#!/usr/bin/env python3
"""Show single-person webcam inference as a live mesh and timing plots."""

import argparse
import socket
import struct
import subprocess
import sys
import time
from collections import deque
from contextlib import ExitStack, contextmanager, suppress
from itertools import islice
from pathlib import Path
from typing import NamedTuple

from benchmark import instrument_session

import cv2
import numpy as np
import onnxruntime as ort
import rerun as rr
import rerun.blueprint as rrb

from instanthmr import HMRPrediction, JOINT_NAMES, MODELS, PosePipeline

LEFT_HIP = JOINT_NAMES.index("left_hip")
RIGHT_HIP = JOINT_NAMES.index("right_hip")


class LiveGeometry(NamedTuple):
    vertices: np.ndarray
    joints_3d: np.ndarray
    joints_2d_normalized: np.ndarray


def output_geometry(person: HMRPrediction, *, camera_space=False, mirror_x=False) -> LiveGeometry:
    """Return the preview geometry in metres and keypoints normalized to the input image."""
    if camera_space:
        vertices, joints = person.vertices_cam, person.joints_3d_cam
    else:
        local = person.joints_3d_local
        pelvis = (local[LEFT_HIP] + local[RIGHT_HIP]) * 0.5
        vertices, joints = person.vertices_local - pelvis, local - pelvis

    height, width = person.image_shape
    image_joints = person.joints_2d / np.array([width, height], dtype=np.float32)
    if mirror_x:
        vertices[:, 0] *= -1
        joints[:, 0] *= -1
        image_joints[:, 0] = 1 - image_joints[:, 0]
    return LiveGeometry(vertices, joints, image_joints)


def output_faces(faces, *, mirror_x=False):
    """Keep the triangle normals facing outward after reflection."""
    return np.ascontiguousarray(faces[:, ::-1]) if mirror_x else faces


def osc_string(value):
    data = value.encode("ascii") + b"\0"
    return data + b"\0" * (-len(data) % 4)


@contextmanager
def open_osc(host, port):
    landmarks = osc_string("/instanthmr/landmarks") + osc_string("," + "f" * (len(JOINT_NAMES) * 3))
    tracking = osc_string("/instanthmr/tracking") + osc_string(",i")
    try:
        family, kind, protocol, _, address = socket.getaddrinfo(host, port, type=socket.SOCK_DGRAM)[0]
        sock = socket.socket(family, kind, protocol)
    except OSError as error:
        raise SystemExit(f"Could not set up OSC for {host}:{port}: {error}") from error

    with sock:
        was_tracked = None

        def send(joints):
            nonlocal was_tracked
            tracked = joints is not None
            try:
                if tracked != was_tracked:
                    sock.sendto(tracking + struct.pack(">i", tracked), address)
                    was_tracked = tracked
                if tracked:
                    sock.sendto(landmarks + joints.astype(">f4", copy=False).tobytes(), address)
            except OSError as error:
                raise SystemExit(f"Could not send OSC to {host}:{port}: {error}") from error

        try:
            yield send
        finally:
            if was_tracked:
                with suppress(OSError):
                    sock.sendto(tracking + struct.pack(">i", 0), address)


@contextmanager
def open_camera(index):
    capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    try:
        if not capture.isOpened():
            raise SystemExit(
                f"Could not open camera {index}. Run from Terminal and allow camera access in "
                "System Settings → Privacy & Security → Camera. For a second camera, try --camera 1."
            )
        yield capture
    finally:
        capture.release()


def camera_frames(capture):
    start = time.perf_counter()
    while True:
        ok, bgr = capture.read()
        if not ok:
            raise SystemExit("The camera stopped delivering frames. Check the camera connection and access.")
        yield time.perf_counter() - start, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def infer_frames(pipeline, frames):
    """Yield the timestamp, upstream prediction (or None), and stage timings for each frame."""
    elapsed = {"pose_onnx_ms": 0.0, "mesh_onnx_ms": 0.0}
    network_run, body_run = pipeline.hmr.network.run, pipeline.hmr.body.run
    instrument_session(pipeline.hmr.network, "pose_onnx_ms", elapsed)
    instrument_session(pipeline.hmr.body, "mesh_onnx_ms", elapsed)
    try:
        for index, (timestamp, rgb) in enumerate(frames):
            elapsed.update(pose_onnx_ms=0.0, mesh_onnx_ms=0.0)
            start = time.perf_counter()
            result = pipeline(rgb, timestamp)
            pipeline_ms = (time.perf_counter() - start) * 1000
            timings = {**elapsed, "pipeline_ms": pipeline_ms}
            if index % pipeline.detector_stride == 0:
                timings["detector_ms"] = result.detector_ms
            yield timestamp, next(iter(result.persons), None), timings
    finally:
        pipeline.hmr.network.run = network_run
        pipeline.hmr.body.run = body_run


@contextmanager
def open_viewer(faces, *, camera_space=False, mirror_x=False):
    eye_z, target_z = (0, 3) if camera_space else (-3, 0)
    view_name = "Camera-space mesh" if camera_space else "Hip-centred mesh"
    if mirror_x:
        view_name += " (mirrored)"
    blueprint = rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name=view_name, line_grid=False,
                          eye_controls=rrb.EyeControls3D(position=[0, 0, eye_z], look_target=[0, 0, target_z],
                                                         eye_up=[0, -1, 0], kind=rrb.Eye3DKind.Orbital)),
        rrb.Vertical(rrb.TimeSeriesView(origin="latency", name="Inference latency (ms)"),
                     rrb.TimeSeriesView(origin="rate", name="Live output (FPS)")),
        column_shares=[2, 1],
    ), collapse_panels=True)
    rr.init("InstantHMR Live")
    stream = rr.binary_stream()
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("world/body", rr.Mesh3D.from_fields(triangle_indices=output_faces(faces, mirror_x=mirror_x),
                                             albedo_factor=[90, 170, 232]),
           static=True)
    # Pipe inference results directly into the viewer without opening a network listener.
    executable = Path(sys.executable).with_name("rerun")
    with subprocess.Popen([str(executable), "--memory-limit", "512MB", "-"], stdin=subprocess.PIPE) as viewer:
        def publish():
            if viewer.poll() is not None:
                raise BrokenPipeError
            if data := stream.read(flush_timeout_sec=5):
                viewer.stdin.write(data)
                viewer.stdin.flush()

        try:
            yield publish
        finally:
            viewer.terminate()
            try:
                viewer.wait(timeout=2)
            except subprocess.TimeoutExpired:
                viewer.kill()


def run_live(pipeline, frames, publish=None, *, camera_space=False, mirror_x=False, send_osc=None):
    completions = deque(maxlen=30)
    last_status = float("-inf")
    had_person = False
    with_inference = infer_frames(pipeline, frames)
    try:
        for timestamp, person, timings in with_inference:
            now = time.perf_counter()
            completions.append(now)
            fps = (len(completions) - 1) / (now - completions[0]) if len(completions) > 1 else 0.0
            geometry = None
            if person is not None and (publish or send_osc):
                geometry = output_geometry(person, camera_space=camera_space, mirror_x=mirror_x)
            if send_osc:
                send_osc(geometry.joints_3d if geometry is not None else None)
            if publish:
                rr.set_time("camera_time", duration=timestamp)
                if geometry is not None:
                    rr.log("world/body", rr.Mesh3D.from_fields(vertex_positions=geometry.vertices))
                elif had_person:
                    rr.log("world/body", rr.Clear(recursive=False))
                for stage, ms in timings.items():
                    rr.log(f"latency/{stage}", rr.Scalars(ms))
                rr.log("rate/output", rr.Scalars(fps))
                publish()
            had_person = person is not None
            if now - last_status >= 1:
                status = "mesh ready" if had_person else "waiting for a person"
                print(f"{fps:5.1f} FPS | pipeline {timings['pipeline_ms']:5.1f} ms | "
                      f"pose {timings['pose_onnx_ms']:5.1f} ms | mesh {timings['mesh_onnx_ms']:5.1f} ms | "
                      f"{status}", flush=True)
                last_status = now
    finally:
        with_inference.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--model", choices=MODELS, default="nano")
    parser.add_argument("--fit", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--device", choices=("cpu", "coreml"), default="cpu")
    parser.add_argument("--detector-stride", type=int, default=2, help="run detection every N frames (default: 2)")
    parser.add_argument("--pose-filter", choices=("one-euro", "alpha-beta"))
    parser.add_argument("--camera-space", action="store_true", help="keep estimated position and depth from the camera")
    parser.add_argument("--mirror-x", action="store_true", help="reflect the mesh horizontally")
    parser.add_argument("--osc-port", type=int, help="enable OSC and send to this UDP port")
    parser.add_argument("--osc-host", default="127.0.0.1", help="OSC receiver address (default: 127.0.0.1)")
    parser.add_argument("--list-landmarks", action="store_true", help="print the landmark indices and exit")
    parser.add_argument("--frames", type=int, help="stop after this many processed frames")
    parser.add_argument("--headless", action="store_true", help="disable the 3D viewer")
    args = parser.parse_args()
    if args.list_landmarks:
        for index, name in enumerate(JOINT_NAMES):
            print(f"{index:2d}  {name}")
        return
    if args.camera < 0 or args.detector_stride < 1 or (args.frames is not None and args.frames < 1):
        parser.error("use a nonnegative camera index and positive frame and stride values")
    if args.osc_port is not None and not 1 <= args.osc_port <= 65535:
        parser.error("--osc-port must be between 1 and 65535")
    if args.device == "coreml" and "CoreMLExecutionProvider" not in ort.get_available_providers():
        parser.error("Core ML is unavailable in this ONNX Runtime build. Use --device cpu.")

    print(f"Loading {args.model} fit {args.fit} for one person…", flush=True)
    pipeline = PosePipeline(args.model, fit=args.fit, device=args.device, detector="nano", max_persons=1,
                            detector_stride=args.detector_stride, pose_filter=args.pose_filter)
    if args.device == "coreml" and pipeline.hmr.provider != "CoreMLExecutionProvider":
        raise SystemExit("The pose session fell back to CPU. Use --device cpu to run that path.")

    try:
        with open_camera(args.camera) as capture, ExitStack() as outputs:
            print(f"Camera {args.camera} is live. Press Ctrl+C in Terminal to stop.", flush=True)
            send_osc = None
            if args.osc_port is not None:
                send_osc = outputs.enter_context(open_osc(args.osc_host, args.osc_port))
                print(f"OSC landmarks → {args.osc_host}:{args.osc_port}", flush=True)
            publish = None
            if not args.headless:
                publish = outputs.enter_context(open_viewer(pipeline.hmr.faces, camera_space=args.camera_space,
                                                            mirror_x=args.mirror_x))
            frames = islice(camera_frames(capture), args.frames)
            run_live(pipeline, frames, publish, camera_space=args.camera_space, mirror_x=args.mirror_x,
                     send_osc=send_osc)
    except (KeyboardInterrupt, BrokenPipeError):
        print("\nCamera stopped.")


if __name__ == "__main__":
    main()
