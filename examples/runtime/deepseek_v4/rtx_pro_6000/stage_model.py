"""Hold a node-local checkpoint in sealed Linux memfds for the life of this process."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
import signal
import threading
from pathlib import Path


def copy_shard(source, destination):
    expected = source.stat().st_size
    fd = os.memfd_create(source.name, os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb", closefd=False) as dst:
            shutil.copyfileobj(src, dst, length=4 << 20)
        if os.fstat(fd).st_size != expected:
            raise OSError(f"Incomplete copy of {source}")
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals | fcntl.F_SEAL_SEAL)
        (destination / source.name).symlink_to(f"/proc/{os.getpid()}/fd/{fd}")
    except BaseException:
        os.close(fd)
        raise
    # Keep this descriptor open until the staging process exits.
    return expected


def stage(source, destination, headroom_gib):
    index = json.loads((source / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values())
    if not names or any(
        Path(n).name != n or not n.endswith(".safetensors") for n in names
    ):
        raise ValueError("Expected safetensors shards in the checkpoint root")
    shards = sorted(source / name for name in names)
    total = sum(shard.stat().st_size for shard in shards)
    available = next(
        line
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if int(available.split()[1]) * 1024 < total + (headroom_gib << 30):
        raise MemoryError("Insufficient available host RAM for checkpoint and headroom")

    # Refuse to reuse an existing directory, including stale links from another PID.
    destination.mkdir(mode=0o700)
    try:
        for path in source.iterdir():
            if path.is_file() and path.suffix in {".json", ".py", ".txt", ".model"}:
                shutil.copyfile(path, destination / path.name)
        if (source / "encoding").is_dir():
            shutil.copytree(source / "encoding", destination / "encoding")
        copied = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            pending = [pool.submit(copy_shard, shard, destination) for shard in shards]
            for count, future in enumerate(concurrent.futures.as_completed(pending), 1):
                copied += future.result()
                print(
                    f"{count}/{len(shards)} shards, {copied / 2**30:.1f} GiB staged",
                    flush=True,
                )

        stopped = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stopped.set())
        (destination / "STAGED").touch()
        print(f"Ready. Set MODEL_PATH={destination}", flush=True)
        stopped.wait()
    finally:
        shutil.rmtree(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "destination", type=Path, help="New directory in private node-local scratch"
    )
    parser.add_argument("--headroom-gib", type=int, default=160)
    args = parser.parse_args()
    if args.headroom_gib < 0:
        parser.error("--headroom-gib must be nonnegative")
    stage(args.source, args.destination, args.headroom_gib)
