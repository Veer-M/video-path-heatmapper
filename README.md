# Video path heatmapper

This finds a red marker in a top-down video and saves its path, a density heatmap, and a CSV of detected positions. A generated sample video makes it easy to try without gameplay footage.

## Run it

Use Python 3.11 or newer. Run these commands from this repository's folder.

```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
```

```sh
python sample_video.py
python heatmapper.py gameplay.avi --output results
python heatmapper.py gameplay.mp4 --min-area 20
```

## How it works and current limits

Tracks the largest red region in each frame, including both HSV red ranges. Multiple similarly sized red players can cause identity switches. Camera movement and red scenery are not compensated for. Density counts detections per frame, so comparisons assume the same video frame rate.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests generate a small video and check marker detection, missing detections, and saved path output.
