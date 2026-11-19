"""Hold a node-local checkpoint in sealed RAM and optionally local disk."""

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import shutil
import signal
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path


def copy_file(source, output, stopped):
    with source.open("rb") as src:
        while chunk := src.read(4 << 20):
            if stopped.is_set():
                raise InterruptedError("Checkpoint staging interrupted")
            output.write(chunk)


def copy_shard(source, destination, disk_directory, stopped):
    expected = source.stat().st_size
    if disk_directory is not None:
        target = disk_directory / source.name
        with target.open("xb") as dst:
            copy_file(source, dst, stopped)
        if target.stat().st_size != expected:
            raise OSError(f"Incomplete copy of {source}")
        (destination / source.name).symlink_to(target)
        return expected

    fd = os.memfd_create(source.name, os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
    try:
        with os.fdopen(fd, "wb", closefd=False) as dst:
            copy_file(source, dst, stopped)
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


def stage(source, destination, headroom_gib, disk_dirs=()):
    index = json.loads((source / "model.safetensors.index.json").read_text())
    names = set(index["weight_map"].values())
    if not names or any(
        Path(n).name != n or not n.endswith(".safetensors") for n in names
    ):
        raise ValueError("Expected safetensors shards in the checkpoint root")
    shards = sorted(source / name for name in names)
    total = sum(shard.stat().st_size for shard in shards)
    remaining = [budget for _, budget in disk_dirs]
    placements = []
    for shard in shards:
        size = shard.stat().st_size
        disk_index = next(
            (i for i, budget in enumerate(remaining) if size <= budget), None
        )
        placements.append(disk_index)
        if disk_index is not None:
            remaining[disk_index] -= size
    disk_bytes = [budget - left for (_, budget), left in zip(disk_dirs, remaining)]
    ram_bytes = total - sum(disk_bytes)

    # Roots may share a filesystem. Check their combined planned use.
    filesystems = {}
    for (root, _), size in zip(disk_dirs, disk_bytes):
        if not root.is_dir():
            raise NotADirectoryError(root)
        device = root.stat().st_dev
        free, used = filesystems.get(device, (shutil.disk_usage(root).free, 0))
        filesystems[device] = (free, used + size)
    if any(used > free for free, used in filesystems.values()):
        raise OSError("Insufficient free disk space for planned checkpoint shards")

    available = next(
        line
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if int(available.split()[1]) * 1024 < ram_bytes + (headroom_gib << 30):
        raise MemoryError("Insufficient available host RAM for checkpoint and headroom")

    with ExitStack() as cleanup:
        stopped = threading.Event()
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: stopped.set())
        cleanup.callback(signal.signal, signal.SIGTERM, previous_handler)
        # Refuse to reuse a directory, including stale links from another PID.
        destination.mkdir(mode=0o700)
        cleanup.callback(shutil.rmtree, destination)
        local_dirs = [
            Path(
                cleanup.enter_context(
                    tempfile.TemporaryDirectory(prefix="sglang-weights-", dir=root)
                )
            )
            if size
            else None
            for (root, _), size in zip(disk_dirs, disk_bytes)
        ]
        for path in source.iterdir():
            if path.is_file() and path.suffix in {".json", ".py", ".txt", ".model"}:
                shutil.copyfile(path, destination / path.name)
        if (source / "encoding").is_dir():
            shutil.copytree(source / "encoding", destination / "encoding")
        print(
            f"Staging {total / 2**30:.1f} GiB "
            f"({ram_bytes / 2**30:.1f} GiB RAM, {sum(disk_bytes) / 2**30:.1f} GiB disk)",
            flush=True,
        )
        copied = 0
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                pending = [
                    pool.submit(
                        copy_shard,
                        shard,
                        destination,
                        local_dirs[index] if index is not None else None,
                        stopped,
                    )
                    for shard, index in zip(shards, placements)
                ]
                for count, future in enumerate(
                    concurrent.futures.as_completed(pending), 1
                ):
                    copied += future.result()
                    print(
                        f"{count}/{len(shards)} shards, {copied / 2**30:.1f} GiB staged",
                        flush=True,
                    )
        except InterruptedError:
            if stopped.is_set():
                return
            raise

        (destination / "STAGED").touch()
        print(f"Ready. Set MODEL_PATH={destination}", flush=True)
        stopped.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "destination", type=Path, help="New directory in private node-local scratch"
    )
    parser.add_argument("--headroom-gib", type=int, default=160)
    parser.add_argument(
        "--disk-dir",
        nargs=2,
        action="append",
        default=[],
        metavar=("PATH", "GIB"),
        help="Stage whole shards within this local disk budget; repeat for more roots",
    )
    args = parser.parse_args()
    if args.headroom_gib < 0:
        parser.error("--headroom-gib must be nonnegative")
    disk_dirs = []
    for root, gib in args.disk_dir:
        try:
            budget = float(gib) * 2**30
        except ValueError:
            parser.error("--disk-dir GIB must be a positive finite number")
        if not math.isfinite(budget) or budget <= 0:
            parser.error("--disk-dir GIB must be a positive finite number")
        disk_dirs.append((Path(root).resolve(), int(budget)))
    stage(args.source, args.destination, args.headroom_gib, disk_dirs)
