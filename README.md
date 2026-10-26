# Video path heatmapper

This program follows colored objects in a top-down video and shows where they move. Each object gets an ID and its own trail. It also measures distance and speed and makes a heatmap that shows where the objects spend time. Everything runs on your computer without a downloaded AI model or GPU.

![Two objects with separate trails and a heatmap of where they spend time](examples/demo.png)

## How it follows an object

The program changes each frame into HSV color space. HSV separates color from brightness, which makes it easier to find red, green, or blue markers. It removes small bits of noise, then groups connected pixels into shapes. The center of each shape is called its centroid. That is the point the program follows.

To keep the same ID between frames, it predicts where an object will move using its last two positions. It matches nearby detections to those predictions. Each detection can match only one track in a frame. If a marker disappears briefly, the program keeps its track available for a set number of missing frames.

This method works best with clear markers and a fixed camera. Objects can swap IDs if they cross or overlap. Changes in lighting, video compression, camera movement, and background objects with the same color can also affect tracking.

## The sample video

The included sample has two red markers moving along different paths. One disappears for six frames, and a tiny red dot acts as background noise. The program finds two tracks and 354 detections across 180 frames. With `--annotated`, it also saves a video that shows the boxes, IDs, and trails.

## Files you get

| File | What it shows |
| --- | --- |
| `heat_path.png` | White dots at all detected positions |
| `tracks.png` | A separate colored trail for each ID |
| `heat_density.png` | A heatmap of where objects spend time |
| `overlay.png` | The heatmap placed over the first frame |
| `positions.csv` | Frame number, time, ID, position, and object area |
| `occupancy_seconds.npy` | The unsmoothed time totals for each pixel |
| `summary.json` | Settings, detection counts, and movement measurements |
| `annotated.avi` | A video with IDs and trails when you use `--annotated` |

The heatmap adds time at the center of each detected object, not across its whole shape. Each detected frame adds `1 / fps` seconds to that pixel. The picture is smoothed and scaled separately for each run. Use the numerical occupancy files if you want to compare time totals between videos.

Distance is the sum of straight lines between an object's recorded positions. Mean speed is that distance divided by the time from the first position to the last, including gaps. Missing positions are not filled in. `observed_seconds` counts frames where the object was found. `span_seconds` is the time between its first and last recorded positions. Speed is in pixels per second unless you add your own conversion to real-world distance.

## Try it

Use Python 3.11 or newer. Open a terminal in this project's folder and create a virtual environment. This keeps its packages separate from your other Python projects.

```sh
python -m venv .venv
```

On Windows, activate it with this command.

```sh
.venv\Scripts\activate
```

On macOS or Linux, use this command instead.

```sh
source .venv/bin/activate
```

Then install the packages the project needs.

```sh
python -m pip install -r requirements.txt
```

```sh
python sample_video.py
python heatmapper.py gameplay.avi --min-area 30 --annotated
```

Open `results/annotated.avi` to watch the sample result. To use your own footage, pass its file name instead.

```sh
python heatmapper.py gameplay.mp4 --color red --min-area 30 --max-area 500
python heatmapper.py gameplay.mp4 --roi 50 20 640 480 --max-distance 15 --max-gap 8 --annotated
python heatmapper.py gameplay.mp4 --max-frames 300 --output short-run
```

`--min-area` and `--max-area` filter shapes by their pixel area. `--roi X Y WIDTH HEIGHT` limits the search to a rectangle inside the frame. Positions in the output still use the full frame's coordinates.

`--max-distance` controls how far a detection can be from its predicted position per elapsed frame. The allowed distance grows after missing frames. Lower it if nearby objects swap IDs. Raise it if fast objects keep getting new IDs. `--max-gap` controls how many missing frames a track can survive.

The program normally reads the frame rate from the video. Use `--fps 30` if that information is missing or wrong. This changes the timestamps and speed calculations too. Processing stops when the video ends or the decoder cannot read another frame. The report covers only the frames that were read.

`tracking.py` handles detection, IDs, and measurements. `heatmapper.py` reads the video and saves the results. `sample_video.py` makes the sample footage.

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests check colors, noise filtering, crop bounds, IDs after missing frames, speed calculations, and videos with no detections. A full test also reads a generated video and checks the CSV, JSON, time totals, and annotated video. GitHub Actions runs the same tests.
