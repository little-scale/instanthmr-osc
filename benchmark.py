#!/usr/bin/env python3
"""Measure InstantHMR on this Mac and optionally save a Rerun recording."""

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import version
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "upstream"
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("RF_HOME", str(ROOT / ".rf-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(UPSTREAM))

import cv2
import numpy as np
import onnxruntime as ort

from instanthmr import MODELS, PosePipeline
from instanthmr.detector import VARIANTS
from instanthmr.visualizer import RerunVisualizer, draw_overlay


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="image to repeat (default: bundled pose sample)")
    source.add_argument("--video", type=Path, help="video to process from the beginning")
    parser.add_argument("--model", choices=MODELS, default="nano")
    parser.add_argument("--fit", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--device", choices=("cpu", "coreml"), default="cpu",
                        help="ONNX Runtime backend for the pose graph (default: cpu)")
    parser.add_argument("--detector", choices=VARIANTS, default="nano")
    parser.add_argument("--max-persons", type=int, default=4)
    parser.add_argument("--detector-stride", type=int, default=1)
    parser.add_argument("--pose-filter", choices=("one-euro", "alpha-beta"))
    parser.add_argument("--frames", type=int, help="frame limit including warm-up (image: 60, video: 300)")
    parser.add_argument("--warmup", type=int, default=5, help="initial frames excluded from statistics")
    parser.add_argument("--out", type=Path, required=True, help="new output directory for this run")
    parser.add_argument("--record", action="store_true", help="also save a 3D recording and a preview image")
    args = parser.parse_args()
    if args.frames is None:
        args.frames = 300 if args.video else 60
    if not 0 <= args.warmup < args.frames:
        parser.error("require 0 <= --warmup < --frames")
    if min(args.max_persons, args.detector_stride) < 1:
        parser.error("--max-persons and --detector-stride must be positive")
    if args.pose_filter and not args.video:
        parser.error("--pose-filter requires --video")
    return args


def video_frames(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise SystemExit(f"Could not open video {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise SystemExit(f"Could not read the video frame rate from {path}")
        index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            yield index / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()


def image_frames(path, count):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"Could not read image {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    for i in range(count):
        yield i / 30, rgb


def instrument_session(session, stage, elapsed):
    run = session.run

    def timed_run(*args, **kwargs):
        start = time.perf_counter()
        result = run(*args, **kwargs)
        elapsed[stage] += (time.perf_counter() - start) * 1000
        return result

    session.run = timed_run


def stats(values):
    if not values:
        return None
    return {
        "samples": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    args = parse_args()
    if args.device == "coreml" and "CoreMLExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("This ONNX Runtime build lacks Core ML support. Try --device cpu.")
    if args.out.exists():
        raise SystemExit(f"Output directory already exists: {args.out}")

    path = (args.video or args.image or ROOT / "samples/pose.jpg").resolve()
    if not path.is_file():
        raise SystemExit(f"Input file does not exist: {path}")
    frames = video_frames(path) if args.video else image_frames(path, args.frames)

    print(f"Loading {args.model} fit {args.fit} ({args.device}) and RF-DETR {args.detector}…", flush=True)
    start = time.perf_counter()
    pipeline = PosePipeline(args.model, fit=args.fit, device=args.device, detector=args.detector,
                            max_persons=args.max_persons, detector_stride=args.detector_stride,
                            pose_filter=args.pose_filter)
    startup_s = time.perf_counter() - start
    if args.device == "coreml" and pipeline.hmr.provider != "CoreMLExecutionProvider":
        raise SystemExit("The pose session fell back to CPU. Use --device cpu to benchmark that path.")

    args.out.mkdir(parents=True, exist_ok=False)
    viewer = RerunVisualizer(pipeline.hmr.faces, spawn=False,
                             save_path=str(args.out / "pose.rrd")) if args.record else None
    elapsed = {"pose_onnx_ms": 0.0, "mesh_onnx_ms": 0.0}
    instrument_session(pipeline.hmr.network, "pose_onnx_ms", elapsed)
    instrument_session(pipeline.hmr.body, "mesh_onnx_ms", elapsed)

    rows = []
    preview_saved = False
    try:
        for i, (timestamp, rgb) in enumerate(islice(frames, args.frames)):
            if i == 0:
                height, width = rgb.shape[:2]
            elapsed.update(pose_onnx_ms=0.0, mesh_onnx_ms=0.0)
            start = time.perf_counter()
            result = pipeline(rgb, timestamp)
            pipeline_ms = (time.perf_counter() - start) * 1000
            rows.append({"frame": i, "timestamp_s": timestamp, "warmup": i < args.warmup,
                         "persons": len(result.persons), "detector_ran": i % args.detector_stride == 0,
                         "detector_ms": result.detector_ms, "hmr_ms": result.hmr_ms,
                         **elapsed, "pipeline_ms": pipeline_ms})
            if args.video and (i + 1) % 30 == 0:
                print(f"Processed {i + 1} video frames…", flush=True)
            if viewer and (args.video or i == 0 or not preview_saved):
                timings = {"detector": result.detector_ms, "InstantHMR": result.hmr_ms}
                viewer.log(i, rgb, result.persons, timings)
                if result.persons and not preview_saved:
                    preview = draw_overlay(rgb, result.persons, timings)
                    preview_saved = cv2.imwrite(str(args.out / "preview.jpg"),
                                                 cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    finally:
        frames.close()
        if viewer:
            viewer.rr.get_global_data_recording().flush()

    if not rows:
        raise SystemExit("No frames could be decoded.")
    with (args.out / "frames.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    measured = [row for row in rows if not row["warmup"]]
    if not measured:
        raise SystemExit("The clip ended during warm-up. Rerun with a smaller --warmup.")
    detections = Counter(row["persons"] for row in measured)
    positive = [row for row in measured if row["persons"]]
    latency = {
        "pipeline_ms": stats([row["pipeline_ms"] for row in measured]),
        "detector_ms": stats([row["detector_ms"] for row in measured if row["detector_ran"]]),
        "hmr_ms": stats([row["hmr_ms"] for row in positive]),
        "pose_onnx_ms": stats([row["pose_onnx_ms"] for row in positive]),
        "mesh_onnx_ms": stats([row["mesh_onnx_ms"] for row in positive]),
    }
    summary = {
        "input": str(path),
        "input_kind": "video" if args.video else "repeated image",
        "input_size_px": {"width": width, "height": height},
        "settings": {k: v for k, v in vars(args).items() if k not in {"image", "video", "out"}},
        "platform": platform.platform(),
        "upstream_commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
                                                   text=True).strip(),
        "versions": {pkg: version(pkg) for pkg in ("onnxruntime", "rfdetr", "torch", "rerun-sdk")},
        "pose_providers": pipeline.hmr.network.get_providers(),
        "mesh_providers": pipeline.hmr.body.get_providers(),
        "detector_device": str(pipeline.detector.model.model.device),
        "startup_s": startup_s,
        "measured_frames": len(measured),
        "frames_by_person_count": dict(sorted(detections.items())),
        "pipeline_fps": 1000 / latency["pipeline_ms"]["mean"],
        "latency_ms": latency,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Pose {summary['pose_providers'][0]}, mesh {summary['mesh_providers'][0]}, "
          f"detector {summary['detector_device']}. Startup {startup_s:.1f} s.")
    print(f"{len(measured)} measured frames, {len(positive)} with people. "
          f"{summary['pipeline_fps']:.1f} pipeline FPS.")
    print(f"{'Stage':<17} {'p50 ms':>10} {'p95 ms':>10} {'Frames':>8}")
    for stage, metrics in latency.items():
        if metrics:
            print(f"{stage:<17} {metrics['p50']:10.2f} {metrics['p95']:10.2f} {metrics['samples']:8}")
    if not positive:
        print("No people found in measured frames. The pose and mesh models were skipped on those frames.")
    print(f"Saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
