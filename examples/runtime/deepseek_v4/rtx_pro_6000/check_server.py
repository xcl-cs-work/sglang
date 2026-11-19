#!/usr/bin/env python3
"""Sample a multi-turn chat and optionally measure concurrent output throughput."""

import argparse
import concurrent.futures
import json
import re
import statistics
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

SUPPORT_TURNS = [
    (
        "I'm on support today. Since 09:10 UTC, some EU customers have had to "
        "submit a PDF export twice. The dashboard still loads normally. Give me "
        "a brief internal ticket title and one question to ask engineering."
    ),
    (
        "Small correction from the incident lead: it's US East, not EU. The first "
        "export sometimes sits in Queued for up to 12 minutes and then finishes "
        "on its own. Submitting it again can create duplicate files, so please "
        "tell people to wait rather than retry. What's a short reply I can send "
        "customers now?"
    ),
    (
        "I'm handing this off to the next shift. The next update is due at "
        "10:00 UTC, and we still don't have a confirmed cause. Write three short "
        "bullets they can paste into the handoff: impact, customer advice, and "
        "next update. Use the latest details and keep times in HH:MM UTC format."
    ),
]
TOPICS = [
    "saving water",
    "cooking vegetables",
    "urban gardens",
    "hiking safely",
    "studying languages",
    "organizing books",
    "birdwatching",
    "taking photographs",
]


def request(args, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    return urlopen(
        Request(
            args.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        ),
        timeout=args.timeout,
    )


def conversation(args):
    with request(args, "/models") as response:
        models = [model["id"] for model in json.load(response)["data"]]
    if args.model not in models:
        raise RuntimeError(f"Requested model {args.model!r} is absent: {models}")

    messages = []
    for turn, prompt in enumerate(SUPPORT_TURNS, 1):
        messages.append({"role": "user", "content": prompt})
        start = time.monotonic()
        with request(
            args,
            "/chat/completions",
            {
                "model": args.model,
                "messages": messages,
                "temperature": 0.7,
                "top_p": 0.9,
                "reasoning_effort": args.reasoning_effort,
                "max_tokens": 640,
            },
        ) as response:
            result = json.load(response)
        choice = result["choices"][0]
        content = choice["message"].get("content")
        if not content or not content.strip():
            raise RuntimeError(f"Turn {turn} returned no visible answer")
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"Turn {turn} exhausted its output token budget")
        messages.append({"role": "assistant", "content": content})
        print(
            json.dumps(
                {
                    "turn": turn,
                    "answer": content,
                    "seconds": round(time.monotonic() - start, 3),
                    "usage": result.get("usage"),
                }
            ),
            flush=True,
        )

    # A smoke check for the explicit correction and unchanged incident context.
    expected = [r"\bUS[\s-]+East\b", r"\b09:10\b", r"\b12\b", r"\b10:00\b"]
    if not all(re.search(pattern, content, re.IGNORECASE) for pattern in expected):
        raise RuntimeError(
            "Final handoff omitted the region, start time, delay, or update"
        )
    if re.search(r"\bEU\b", content):
        raise RuntimeError("Final handoff retained the superseded EU region")
    print(json.dumps({"conversation": "passed"}), flush=True)


def stream_sample(args, topic, gate, limit):
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": f"Write 50 practical tips about {topic}. "
                "Give a full sentence for every tip.",
            }
        ],
        "temperature": 0.7,
        "reasoning_effort": "none",
        "max_tokens": limit,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    gate.wait(timeout=args.timeout)
    start = time.monotonic()
    first = last = usage = None
    done = False
    with request(args, "/chat/completions", body) as response:
        for line in response:
            if time.monotonic() - start > args.timeout:
                raise TimeoutError("Stream exceeded the request time budget")
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                done = True
                break
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError(f"Streaming error: {event['error']}")
            if event.get("usage"):
                usage = event["usage"]
            if any(
                choice.get("delta", {}).get("content")
                for choice in event.get("choices", [])
            ):
                last = time.monotonic()
                if first is None:
                    first = last
    end = time.monotonic()
    if not done or first is None or not usage or usage.get("completion_tokens", 0) <= 0:
        raise RuntimeError(
            "Incomplete stream, missing visible output, or missing token usage"
        )
    return {
        "start": start,
        "end": end,
        "output_tokens": usage["completion_tokens"],
        "ttft_seconds": first - start,
        "decode_tokens_per_second": (
            (usage["completion_tokens"] - 1) / (last - first) if last > first else None
        ),
    }


def benchmark(args):
    # Warm up outside the measured batches. Each batch starts together at a barrier.
    stream_sample(args, TOPICS[0], threading.Barrier(1), 16)
    for repetition in range(args.repetitions):
        levels = args.concurrency if repetition % 2 == 0 else args.concurrency[::-1]
        for concurrency in levels:
            gate = threading.Barrier(concurrency)
            with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
                pending = [
                    pool.submit(
                        stream_sample,
                        args,
                        TOPICS[(index + repetition) % len(TOPICS)],
                        gate,
                        args.max_tokens,
                    )
                    for index in range(concurrency)
                ]
                rows = [future.result() for future in pending]
            seconds = max(row["end"] for row in rows) - min(
                row["start"] for row in rows
            )
            tokens = sum(row["output_tokens"] for row in rows)
            decode_rates = [
                row["decode_tokens_per_second"]
                for row in rows
                if row["decode_tokens_per_second"] is not None
            ]
            print(
                json.dumps(
                    {
                        "concurrency": concurrency,
                        "repetition": repetition + 1,
                        "output_tokens": tokens,
                        "seconds": round(seconds, 3),
                        "aggregate_output_tokens_per_second": round(
                            tokens / seconds, 3
                        ),
                        "median_ttft_seconds": round(
                            statistics.median(row["ttft_seconds"] for row in rows), 3
                        ),
                        "median_stream_decode_tokens_per_second": (
                            round(statistics.median(decode_rates), 3)
                            if decode_rates
                            else None
                        ),
                    }
                ),
                flush=True,
            )


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:31041/v1")
    parser.add_argument("--model", default="deepseek-v4.1-flash-1017")
    parser.add_argument(
        "--timeout", type=positive_int, default=180, help="request timeout in seconds"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "high", "xhigh", "max"],
        default="low",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="also measure concurrent streaming output",
    )
    parser.add_argument(
        "--concurrency", type=positive_int, nargs="+", default=[1, 2, 4, 8]
    )
    parser.add_argument("--repetitions", type=positive_int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=128,
        help="output limit per benchmark request",
    )
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    url = urlsplit(args.base_url)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or not url.path.endswith("/v1")
        or url.query
        or url.fragment
        or url.username
        or url.password
    ):
        parser.error(
            "--base-url must be an HTTP(S) URL ending in /v1 without credentials or a query"
        )
    try:
        conversation(args)
        if args.benchmark:
            benchmark(args)
    except (
        HTTPError,
        URLError,
        TimeoutError,
        RuntimeError,
        ValueError,
        KeyError,
        IndexError,
        threading.BrokenBarrierError,
    ) as exc:
        parser.exit(1, f"Check failed: {exc}\n")


if __name__ == "__main__":
    main()
