# InstantHMR OSC

Single-person 3D body tracking from a Mac webcam, with live Open Sound Control (OSC) output for 70 landmarks and an optional 3D mesh viewer. This project uses [InstantHMR 1.1](https://github.com/Naturalpad-studio/InstantHMR) for pose inference and [Rerun](https://rerun.io/) for the preview.

## Setup on Apple Silicon

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/). From Terminal, run:

```sh
git clone https://github.com/little-scale/instanthmr-osc.git
cd instanthmr-osc
git clone https://github.com/Naturalpad-studio/InstantHMR.git upstream
git -C upstream checkout --detach aa4b7df48e1a91aa3b61d8e7e7d7db1f60b6ef52
uv --cache-dir .uv-cache venv --python 3.12 .venv
uv --cache-dir .uv-cache pip install --python .venv/bin/python -r upstream/requirements.txt
```

Run the commands below from the `instanthmr-osc` folder. The first run downloads the selected pose and detector weights. Once they are cached, you can prepend `HF_HUB_OFFLINE=1` to skip Hugging Face network checks.

## Live webcam

```sh
.venv/bin/python live_camera.py
```

Allow camera access for Terminal if macOS asks, then run the command again if the first attempt exits. The Rerun window shows one rotatable body mesh centred between the hips, with live FPS and inference timings. The camera image stays off-screen. The session runs in memory. Close the viewer or press Ctrl+C in Terminal to stop.

The default is the Nano pose model with one crop. RF-DETR detects a person every second processed frame and updates the box from pose keypoints in between. Use `--detector-stride 1` to detect every frame, `--camera 1` for a second camera, or `--headless` for Terminal timings alone. Live FPS measures completed outputs, including camera waits and the viewer handoff. Pipeline latency measures detection, pose inference, and mesh generation. Remove `HF_HUB_OFFLINE=1` when trying an uncached model or fit mode.

Use `--camera-space` to keep the estimated lateral position, height, and depth relative to the input camera. Add `--mirror-x` for a selfie-style horizontal flip in either view mode.

```sh
.venv/bin/python live_camera.py --pose-filter one-euro --camera-space --mirror-x
```

Camera-space 3D coordinates are in metres, with the camera at the origin, X pointing right, Y down, and Z forward. Depth and scale are estimates from a single camera. Mirroring reflects X about the camera centreline in camera space, or about the hips in the default view.

Python consumers can use `live_camera.infer_frames(pipeline, frames)` for each timestamp, native prediction, and timing dictionary. Pass a prediction to `live_camera.output_geometry(prediction, camera_space=True, mirror_x=True)` to get the displayed vertices, matching 3D joints, and 2D keypoints divided by the input width and height. The 2D origin is the top-left corner, and points outside the image can fall outside 0–1. Use `live_camera.output_faces(pipeline.hmr.faces, mirror_x=True)` with mirrored vertices to keep the triangle winding consistent.

### OSC landmarks

Set your OSC receiver to UDP port 9000 and start the webcam with `--osc-port 9000`. The default receiver address is `127.0.0.1`. Use `--osc-host` for a receiver on another machine or `--headless` to stream without the viewer.

```sh
.venv/bin/python live_camera.py --pose-filter one-euro --camera-space --mirror-x --osc-port 9000
.venv/bin/python live_camera.py --list-landmarks
```

| OSC address | Arguments | When sent |
| --- | --- | --- |
| `/instanthmr/landmarks/x` | 70 X coordinates in metres | Each frame with a detected person |
| `/instanthmr/landmarks/y` | 70 Y coordinates in metres | Each frame with a detected person |
| `/instanthmr/landmarks/z` | 70 Z coordinates in metres | Each frame with a detected person |
| `/instanthmr/tracking` | One integer, `1` for tracked or `0` for absent | First frame, tracking changes, and shutdown if tracked |

The three axes arrive as separate UDP packets. Landmark `i` is argument `i` in each axis message. For example, read argument 62 from X, Y, and Z for the left wrist. The OSC coordinates follow `--camera-space` and `--mirror-x`. Mirroring preserves the anatomical landmark names and message order. To send to another machine, add its receiver IP, for example `--osc-host 192.168.1.42 --osc-port 9000`.

To skip the full mesh, install the optional ONNX package and pass `--landmarks-only`:

```sh
uv --cache-dir .uv-cache pip install --python .venv/bin/python onnx
.venv/bin/python live_camera.py --pose-filter one-euro --camera-space --mirror-x --landmarks-only --osc-port 9000
```

This mode computes only the 468 body vertices needed to produce the 70 landmarks during live inference. The viewer shows the 3D landmark points. Add `--headless` for OSC and Terminal timings alone.

## Landmark reference

The table lists all 70 [InstantHMR landmarks](https://github.com/Naturalpad-studio/InstantHMR/blob/aa4b7df48e1a91aa3b61d8e7e7d7db1f60b6ef52/instanthmr/skeleton.py) in OSC message order. The zero-based index is the argument position in each axis message.

| Index | Landmark |
| ---: | --- |
| 0 | `nose` |
| 1 | `left_eye` |
| 2 | `right_eye` |
| 3 | `left_ear` |
| 4 | `right_ear` |
| 5 | `left_shoulder` |
| 6 | `right_shoulder` |
| 7 | `left_elbow` |
| 8 | `right_elbow` |
| 9 | `left_hip` |
| 10 | `right_hip` |
| 11 | `left_knee` |
| 12 | `right_knee` |
| 13 | `left_ankle` |
| 14 | `right_ankle` |
| 15 | `left_big_toe_tip` |
| 16 | `left_small_toe_tip` |
| 17 | `left_heel` |
| 18 | `right_big_toe_tip` |
| 19 | `right_small_toe_tip` |
| 20 | `right_heel` |
| 21 | `right_thumb_tip` |
| 22 | `right_thumb_first_joint` |
| 23 | `right_thumb_second_joint` |
| 24 | `right_thumb_third_joint` |
| 25 | `right_index_tip` |
| 26 | `right_index_first_joint` |
| 27 | `right_index_second_joint` |
| 28 | `right_index_third_joint` |
| 29 | `right_middle_tip` |
| 30 | `right_middle_first_joint` |
| 31 | `right_middle_second_joint` |
| 32 | `right_middle_third_joint` |
| 33 | `right_ring_tip` |
| 34 | `right_ring_first_joint` |
| 35 | `right_ring_second_joint` |
| 36 | `right_ring_third_joint` |
| 37 | `right_pinky_tip` |
| 38 | `right_pinky_first_joint` |
| 39 | `right_pinky_second_joint` |
| 40 | `right_pinky_third_joint` |
| 41 | `right_wrist` |
| 42 | `left_thumb_tip` |
| 43 | `left_thumb_first_joint` |
| 44 | `left_thumb_second_joint` |
| 45 | `left_thumb_third_joint` |
| 46 | `left_index_tip` |
| 47 | `left_index_first_joint` |
| 48 | `left_index_second_joint` |
| 49 | `left_index_third_joint` |
| 50 | `left_middle_tip` |
| 51 | `left_middle_first_joint` |
| 52 | `left_middle_second_joint` |
| 53 | `left_middle_third_joint` |
| 54 | `left_ring_tip` |
| 55 | `left_ring_first_joint` |
| 56 | `left_ring_second_joint` |
| 57 | `left_ring_third_joint` |
| 58 | `left_pinky_tip` |
| 59 | `left_pinky_first_joint` |
| 60 | `left_pinky_second_joint` |
| 61 | `left_pinky_third_joint` |
| 62 | `left_wrist` |
| 63 | `left_olecranon` |
| 64 | `right_olecranon` |
| 65 | `left_cubital_fossa` |
| 66 | `right_cubital_fossa` |
| 67 | `left_acromion` |
| 68 | `right_acromion` |
| 69 | `neck` |

## Run a benchmark

```sh
.venv/bin/python benchmark.py --image "/path/to/photo.jpg" --out results/my-photo
.venv/bin/python benchmark.py --image "/path/to/photo.jpg" --device coreml --out results/my-photo-coreml
.venv/bin/python benchmark.py --video "/path/to/clip.mp4" --out results/my-clip
```

The photo command repeats the input image. The video command reads up to the first 300 frames. Use `--frames` to change the limit and `--warmup` to change how many initial frames are excluded. Every run needs a fresh `--out` directory. Media is processed locally. The first use of a model or detector variant downloads its weights.

The defaults are the `nano` pose model, `--fit 1` for one crop per person, CPU for the pose graph, and the `nano` RF-DETR person detector with a cap of four people. CPU was faster on the bundled single-person photo. `--device coreml` switches the **pose** graph to Core ML, which can still assign some operators to CPU. RF-DETR selects its own PyTorch device, and the mesh graph runs on CPU with either setting. The report records provider order and the detector's device.

Each run saves `summary.json` with p50 and p95 latency, a breakdown of people per frame, and pipeline FPS. `frames.csv` has per-frame values. The pipeline time includes detection, cropping, pose inference, mesh generation, and box updates. It excludes video decoding, recording, and viewer rendering. Pose and mesh ONNX times are components of the HMR stage. Each is a total per frame across the people found. The summary counts detector time only on frames where detection ran and reports HMR timings only for frames with at least one person. Pipeline FPS covers all measured frames.

## Review pose quality

```sh
.venv/bin/python benchmark.py --video "/path/to/clip.mp4" --model nano --fit 1 \
  --record --out results/my-review-nano
.venv/bin/python benchmark.py --video "/path/to/clip.mp4" --model medium --fit 2 \
  --record --out results/my-review-medium
.venv/bin/rerun results/my-review-medium/pose.rrd
```

`--record` saves the 2D skeleton and rotatable 3D mesh in `pose.rrd`, plus a `preview.jpg` from the first frame with a detection. Scrub the same frames in each recording to check joint placement, missed detections, left/right swaps, and jitter. Raise `--max-persons` to evaluate more than four people. Use `--frames` to keep recordings short, since they include full-resolution frames and meshes.

Compare one setting at a time to isolate its effect. `--fit 0`, `1`, and `2` use the pose network alone, a one-crop fit, or a two-crop fit. `--pose-filter one-euro` lets you inspect video jitter and lag. Use the same clip, detector, and person cap, and omit `--record` for speed comparisons to avoid recording overhead. The supplied samples support visual review. For numerical 3D accuracy, the [upstream EMDB-1 and 3DPW evaluation](upstream/benchmark/README.md) requires dataset access and separate MHR and SMPL assets. That evaluation has a CPU option.

The `nano`, `tiny`, `small`, and `medium` weights are for research and non-commercial use. `medium-commercial` is available subject to the [SAM License](upstream/LICENSE-SAM). See the [upstream model notes](https://github.com/Naturalpad-studio/InstantHMR#licence).

## Optional benchmark samples

```sh
mkdir -p samples
curl -fL https://storage.googleapis.com/mediapipe-assets/pose.jpg -o samples/pose.jpg
curl -fL https://user-images.githubusercontent.com/87690686/137440639-fb08603d-9a35-474e-b65f-46b5c06b68d6.mp4 -o samples/motion.mp4
```

The still image is the [MediaPipe pose sample](https://storage.googleapis.com/mediapipe-assets/pose.jpg) and becomes the default image for `benchmark.py` once downloaded. The motion clip comes from the [MMPose whole-body demo](https://github.com/open-mmlab/mmpose/blob/main/demo/docs/en/2d_wholebody_pose_demo.md).
