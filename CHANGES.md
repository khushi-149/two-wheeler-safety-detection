# What changed and why

## Bugs fixed

| # | Problem in the original | Effect | Fix |
|---|---|---|---|
| 1 | `requirements.txt` contained `lap>=0.5.12lap>=0.5.12` | `pip install -r requirements.txt` failed | Corrected; added versions, removed unused `supervision` and `pyyaml` |
| 2 | Rider-count smoothing required 3 agreeing analyses before reporting 3+ riders | **A photo (one frame) could never be reported as triple riding** in the pipeline / dashboard | Photos skip the history requirement (`smoothed_count(..., min_history_for_triple=1)`); videos keep the protection |
| 3 | `with sqlite3.connect()` only commits, it does not close | Leaked a connection on every database call; locks the file on Windows | Connections are always closed; failed writes roll back; added an index |
| 4 | "Could not open source: rtsp://user:PASSWORD@..." | Camera password shown in the dashboard and logs | Credentials are removed from every message (`redact_source`) |
| 5 | Dashboard saved uploads under the client-supplied file name | A crafted name such as `../../x` could write outside the upload folder | Only the base name is used; unchanged files are not rewritten on each click |
| 6 | Helmet class ids hard-coded (0 = with, 1 = without) | A retrained model with another class order would silently swap helmet / no-helmet | Class ids are read from the model's class names, with safe defaults |
| 7 | `cv2.imshow` on a headless OpenCV build | Crash | The run continues without a preview window |
| 8 | Missing `helmet_detector.pt` | Long stack trace | Clear message saying where to get the file |
| 9 | Scripts used paths relative to the current folder | Only worked when started from the project root | Paths anchored to the project root (`settings.py`) |
| 10 | `test_riders.py` defined `test_image` / `test_video`; `test_yolo.py` / `test_tracking.py` ran code on import | `pytest` collected them and failed; importing them started a video run | Renamed / replaced (see below) |
| 11 | `None` video time in the dashboard evidence caption | Formatting error | Guarded |
| 12 | Training script ran on import, with hard-coded settings | Could not be imported or configured | `argparse` + `main()` |

## Structure

* `pipeline.py` (1,390 lines) split into `settings`, `geometry`, `assignment`, `capture`,
  `drawing` and `pipeline`. Logic is unchanged apart from the fixes above.
* Duplicate helpers (`iou` / `box_iou` / `box_ioa`, `box_center`, `point_in_box`) now live once in `geometry.py`.
* All thresholds and paths moved to `settings.py`.
* `print` in library code replaced with `logging`; command-line tools keep their normal output.
* Type hints and docstrings on the public functions; `argparse` for every command-line tool.

## Files removed / renamed

| Original | Now |
|---|---|
| `src/test_riders.py` | `src/interactive_tester.py` (same features) |
| `src/detect_video.py`, `src/detect_video_tracking.py` | Removed: older, hard-coded copies of what `detect_video_full.py` + `pipeline.py` do (tracking, evidence, database, plates) |
| `src/test_yolo.py`, `src/test_tracking.py` | Replaced by `scripts/check_setup.py` |
| `dashboard/` (empty folder) | Removed; the dashboard is `src/dashboard.py` |
| Notebook | Duplicate cells and ~50 KB of install output removed; metrics printed |

## Added

`tests/` (automated tests), `.github/workflows/ci.yml`, `pyproject.toml`, `requirements-dev.txt`,
`scripts/check_setup.py`, folder READMEs, a README that matches the code, and a privacy note.

## Not verified here

The tests use scripted stand-in models, so the logic around the models is tested, but the real
YOLO weights, a GPU, EasyOCR, Streamlit and a live camera were **not** available in the
environment used to prepare these changes. Run `python scripts/check_setup.py --load`, then try one
photo and one video on your machine before relying on it.

## Helmet re-check and mannequin fix

- **Bystander filter** (`assignment.is_plausible_rider`): a person whose head is far above the
  motorcycle box (over 0.80 bike-heights AND over 0.50 person-heights) is not a rider. Stops shop
  mannequins / pedestrians behind a parked scooter from being counted or flagged as triple riding.
  Limits are in `settings.py` (`MAX_HEAD_ABOVE_MOTO`, `MAX_HEAD_ABOVE_PERSON`).
- **Head re-check** (new `src/head_check.py`): a zoomed second look at each rider's head.
  It recovers heads the main pass missed and double-checks a doubtful "With Helmet"
  (below `VERIFY_BELOW`). Settings: `RECOVERY_CONF`, `RECOVERY_IMGSZ`, `VERIFY_*`.
- **Photo-adaptive size**: photos are analysed at the crop's own size (up to `CROP_IMGSZ_MAX`=1280);
  videos keep `CROP_IMGSZ`=640 for speed.
- **`INFER_NO_HELMET`** (default False): set True to flag riders whose head is not found at all.
- Plates are read only for violating bikes and need `models/plate_detector.pt`.
- Tested with scripted fake models only; not yet verified against the real weights.

## Cleaner labels and rider count

- Labels are sized as a share of the picture (no more giant text on 4000 px photos), drawn as small
  non-overlapping chips that stay inside the picture, and placed BELOW the bike so they no longer
  cover the riders' heads. One compact "NO HELMET VIOLATION / TRIPLE RIDING DETECTED" banner.
- Rider count: tiny people far behind the bike (stall visitors, pedestrians) are no longer counted
  as riders or "hidden pillions" (`MIN_PERSON_TO_BIKE`, `MIN_HIDDEN_HEAD_TO_BIKE`,
  `MIN_HIDDEN_HEAD_VS_RIDERS` in settings.py).

## Smooth, aligned video preview + photo accuracy

- **Video preview (`src/display.py`, new):** a video FILE in real-time mode is now shown slightly
  delayed (adapts to your PC's detection speed, 0.35-2 s) so each frame is drawn with the detections
  made on that frame. Boxes stay on the riders instead of trailing behind. Settings:
  `ALIGNED_FILE_DISPLAY`, `DISPLAY_MIN_DELAY`, `DISPLAY_MAX_DELAY`. Cameras keep the lowest-latency view.
- **Bystanders:** a person whose body barely overlaps the bike (sitting on steps behind a parked bike)
  is not a rider (`MIN_RIDER_BODY_OVERLAP`).
- **Photos only:** stronger pose model `yolo11m-pose.pt` (downloads on first photo; finds hidden
  pillions) and a lower helmet confidence (`PHOTO_HELMET_CONF`). A rider whose head cannot be found
  gets an amber "HELMET NOT FOUND" marker for a human to check. It is NOT logged as a violation
  (set `INFER_NO_HELMET = True` to log such riders as "no helmet").
- **`scripts/debug_photo.py` (new):** prints every person / head the pipeline sees on a photo and why each
  was or was not counted.

## Head-to-rider matching fix (crowded riders)

- Heads are now matched to riders globally by distance to each rider's pose head point, instead of
  rider-by-rider. Fixes the "4 riders" on triple_3.jpg, where the rear rider took the middle rider's
  helmet and left the rear rider's own helmet over as a phantom "hidden pillion".
- `scripts/debug_photo.py` now also runs the zoomed head re-check, so its output matches the dashboard.