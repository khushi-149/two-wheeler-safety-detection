"""
Two-Wheeler Safety Monitor - final dashboard.

Run from the project root:
    streamlit run src/dashboard.py

Inputs : video file, photo(s), webcam, CCTV / network camera, mobile phone camera
Outputs: live view + live counters, results per run (violations, plates,
         evidence, processed video/photo), and the full violation history.
"""

import os
import shutil
import subprocess
import threading
from urllib.parse import urlparse

import cv2
import pandas as pd
import streamlit as st

from pipeline import ViolationPipeline, IMAGE_EXTENSIONS
from video_menu import list_videos, find_cameras
from violation_db import ViolationDB

DB_PATH = "outputs/violations.db"
VIDEO_FOLDER = "data/videos"
VIDEO_UPLOADS = "data/videos/uploads"
PHOTO_FOLDER = "data/images"
PHOTO_UPLOADS = "data/images/uploads"
PLATE_MODEL = "models/plate_detector.pt"

for folder in (VIDEO_UPLOADS, PHOTO_UPLOADS):
    os.makedirs(folder, exist_ok=True)

st.set_page_config(page_title="Two-Wheeler Safety Monitor", layout="wide")

db = ViolationDB(DB_PATH)

if "runs" not in st.session_state:
    st.session_state.runs = {}          # run_id -> {"label": ..., "stats": ...}

if "cameras" not in st.session_state:
    st.session_state.cameras = None     # result of the last camera scan


# -------------------- Helpers --------------------

@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_pipeline(use_plates):
    return ViolationPipeline(use_plates=use_plates, db_path=DB_PATH)


def show_image(container, image, **kwargs):
    """st.image that works on both old and new Streamlit versions."""
    try:
        container.image(image, width="stretch", **kwargs)
    except Exception:
        container.image(image, use_container_width=True, **kwargs)


def show_table(container, df):
    try:
        container.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        container.dataframe(df, use_container_width=True, hide_index=True)


def list_photos(folder):
    found = []

    for root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                found.append(os.path.join(root, name))

    return sorted(found)


def save_upload(uploaded, folder):
    path = os.path.join(folder, uploaded.name)

    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    return path


def stream_label(url):
    """Show only the host of a stream URL (never its username/password)."""
    try:
        return urlparse(url).hostname or "stream"
    except ValueError:
        return "stream"


def test_connection(source, timeout=10):
    """Try to grab one frame from a camera/stream. Returns the frame or None."""
    result = {}

    def worker():
        cap = None

        try:
            if isinstance(source, int) and os.name == "nt":
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                if isinstance(source, str) and source.lower().startswith("rtsp"):
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

                cap = cv2.VideoCapture(source)

            if cap.isOpened():
                ok, frame = cap.read()

                if ok:
                    result["frame"] = frame
        finally:
            if cap is not None:
                cap.release()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)

    return result.get("frame")


def ffmpeg_path():
    found = shutil.which("ffmpeg")

    if found:
        return found

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


@st.cache_data(show_spinner="Preparing the video for the browser...")
def browser_video(path, modified):
    """
    OpenCV writes MPEG-4 video that most browsers cannot play. If ffmpeg is
    available, convert to H.264 for in-page playback. Returns None otherwise.
    """
    ffmpeg = ffmpeg_path()

    if not ffmpeg:
        return None

    out = path[:-4] + "_web.mp4"

    if not os.path.exists(out):
        try:
            subprocess.run(
                [ffmpeg, "-y", "-i", path, "-vcodec", "libx264",
                 "-pix_fmt", "yuv420p", "-preset", "veryfast", "-an", out],
                check=True, capture_output=True, timeout=900,
            )
        except Exception:
            return None

    return out


def records_to_frame(records):
    df = pd.DataFrame(records)

    if df.empty:
        return df

    return df.rename(columns={
        "track_id": "bike ID",
        "violation_type": "violation",
        "detected_at": "detected at",
        "video_time": "time (s)",
        "plate_number": "plate",
        "plate_conf": "plate conf",
    })


# -------------------- Sidebar: choose the input --------------------

st.sidebar.title("Input source")

SOURCE_TYPES = [
    "Video file",
    "Photo(s)",
    "Webcam",
    "CCTV / network camera",
    "Mobile phone camera",
]

source_type = st.sidebar.radio("Where is the footage coming from?", SOURCE_TYPES)

# Each job: (source, label, is_live). Filled in below, run when Start is pressed.
jobs = []
ready_hint = None
is_live_mode = False

if source_type == "Video file":

    mode = st.sidebar.radio("Video", ["Choose from data/videos", "Upload a video"])

    if mode == "Choose from data/videos":
        videos = list_videos(VIDEO_FOLDER)

        if videos:
            picked = st.sidebar.selectbox(
                "Test video", videos,
                format_func=lambda p: os.path.relpath(p, VIDEO_FOLDER),
            )
            jobs.append((picked, os.path.basename(picked), False))
        else:
            ready_hint = f"No videos found in {VIDEO_FOLDER}. Upload one instead."
    else:
        uploaded = st.sidebar.file_uploader(
            "Choose a video", type=["mp4", "avi", "mov", "mkv", "webm"]
        )

        if uploaded is not None:
            path = save_upload(uploaded, VIDEO_UPLOADS)
            jobs.append((path, uploaded.name, False))

elif source_type == "Photo(s)":

    mode = st.sidebar.radio("Photos", ["Upload photos", "Choose from data/images"])

    if mode == "Upload photos":
        uploads = st.sidebar.file_uploader(
            "Choose one or more photos",
            type=[e.lstrip(".") for e in IMAGE_EXTENSIONS],
            accept_multiple_files=True,
        )

        for up in uploads or []:
            jobs.append((save_upload(up, PHOTO_UPLOADS), up.name, False))
    else:
        photos = list_photos(PHOTO_FOLDER)

        if photos:
            chosen = st.sidebar.multiselect(
                "Photos", photos,
                format_func=lambda p: os.path.relpath(p, PHOTO_FOLDER),
            )
            for p in chosen:
                jobs.append((p, os.path.basename(p), False))
        else:
            ready_hint = f"No photos found in {PHOTO_FOLDER}. Upload some instead."

elif source_type == "Webcam":

    is_live_mode = True

    if st.sidebar.button("Scan for cameras"):
        with st.spinner("Scanning..."):
            st.session_state.cameras = find_cameras()

    options = st.session_state.cameras or [0, 1, 2, 3]

    if st.session_state.cameras == []:
        st.sidebar.warning("No camera found. Check that it is connected and not "
                           "in use by another app.")

    index = st.sidebar.selectbox("Camera", options, format_func=lambda i: f"Camera {i}")

    if st.sidebar.button("Test camera"):
        frame = test_connection(index)

        if frame is None:
            st.sidebar.error("Could not read from this camera.")
        else:
            show_image(st.sidebar, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                       caption="Camera works")

    st.sidebar.caption(
        "The webcam must be on the same computer that runs this dashboard."
    )
    jobs.append((int(index), f"Webcam {index}", True))

elif source_type == "CCTV / network camera":

    is_live_mode = True

    url = st.sidebar.text_input(
        "Stream URL",
        type="password",
        placeholder="rtsp://user:password@192.168.1.10:554/stream1",
        help="RTSP or HTTP stream of the camera. Hidden while typing because "
             "it often contains a password.",
    ).strip()

    with st.sidebar.expander("Where do I find the URL?"):
        st.markdown(
            "- Look in the camera / DVR / NVR manual or its web page for an "
            "**RTSP URL**.\n"
            "- Typical shape: `rtsp://user:password@IP:554/stream1` "
            "(the path differs between brands).\n"
            "- The computer must be on the same network (or have access to it)."
        )

    if st.sidebar.button("Test connection") and url:
        with st.spinner("Connecting (up to 10 s)..."):
            frame = test_connection(url)

        if frame is None:
            st.sidebar.error("Could not read from this stream. Check the URL, "
                             "login and network.")
        else:
            show_image(st.sidebar, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                       caption="Stream works")

    if url:
        jobs.append((url, f"CCTV {stream_label(url)}", True))
    else:
        ready_hint = "Enter the camera's stream URL."

else:  # Mobile phone camera

    is_live_mode = True

    url = st.sidebar.text_input(
        "Phone camera URL",
        placeholder="http://192.168.1.5:8080/video",
    ).strip()

    with st.sidebar.expander("How do I use my phone as a camera?"):
        st.markdown(
            "1. Install an **IP-camera app** on the phone (for example "
            "*IP Webcam* on Android, or *DroidCam*).\n"
            "2. Connect the phone and this computer to the **same Wi-Fi**.\n"
            "3. Start the app's server. It shows an address such as "
            "`http://192.168.1.5:8080`.\n"
            "4. Paste the video address here. For IP Webcam it is usually the "
            "address followed by `/video`; DroidCam usually uses "
            "`:4747/video`.\n"
            "5. Press **Test connection**, then **Start**."
        )

    if st.sidebar.button("Test connection") and url:
        with st.spinner("Connecting (up to 10 s)..."):
            frame = test_connection(url)

        if frame is None:
            st.sidebar.error("Could not read from the phone. Check the address "
                             "and that both devices share the same Wi-Fi.")
        else:
            show_image(st.sidebar, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                       caption="Phone camera works")

    if url:
        jobs.append((url, f"Phone {stream_label(url)}", True))
    else:
        ready_hint = "Enter the phone camera's video address."

# -------------------- Sidebar: settings + controls --------------------

st.sidebar.divider()
st.sidebar.subheader("Settings")

plate_model_present = os.path.exists(PLATE_MODEL)

use_plates = st.sidebar.checkbox(
    "Read number plates",
    value=plate_model_present,
    help="Needs models/plate_detector.pt",
)

if use_plates and not plate_model_present:
    st.sidebar.caption("models/plate_detector.pt not found: plate reading "
                       "will stay off.")

is_photo_mode = source_type == "Photo(s)"

record_video = False

if not is_photo_mode:
    record_video = st.sidebar.checkbox(
        "Record the annotated video",
        value=not is_live_mode,
        help="Saved in outputs/videos",
    )

update_every = st.sidebar.slider(
    "Refresh the live view every N frames", 1, 10, 2 if is_live_mode else 3
)

start = st.sidebar.button("Start", type="primary", disabled=not jobs)
st.sidebar.button("Stop", help="Ends the current run.")

if ready_hint and not jobs:
    st.sidebar.info(ready_hint)

# -------------------- Main layout --------------------

st.title("Two-Wheeler Safety Monitor")

tab_live, tab_results, tab_history = st.tabs(
    ["Live monitor", "Results", "History"]
)

METRICS = [
    ("Motorcycles", "motorcycles"),
    ("Helmet violations", "helmet_violations"),
    ("Triple riding", "triple_riding"),
    ("Plates read", "plates_read"),
    ("Evidence saved", "evidence_saved"),
]

# -------------------- Live monitor --------------------

with tab_live:

    metric_slots = [col.empty() for col in st.columns(len(METRICS))]

    def render_metrics(stats):
        for slot, (label, key) in zip(metric_slots, METRICS):
            slot.metric(label, stats.get(key, 0))

    render_metrics({})

    col_view, col_side = st.columns([3, 1])

    status_slot = col_view.empty()
    frame_slot = col_view.empty()
    progress_slot = col_view.empty()

    col_side.subheader("Latest violations")
    recent_slot = col_side.empty()

    if not start:
        frame_slot.info(
            "Choose an input source in the sidebar, then press **Start**. "
            "The live view appears here; final results are in the "
            "**Results** tab."
        )

    def make_callback(label, is_live):

        state = {"helmet": 0, "triple": 0, "evidence": -1}
        progress = progress_slot.progress(0.0) if not is_live else None

        def on_frame(frame_bgr, stats):

            run_id = stats["run_id"]
            st.session_state.runs[run_id] = {"label": label, "stats": stats}

            h, w = frame_bgr.shape[:2]

            if w > 960:
                frame_bgr = cv2.resize(frame_bgr, (960, int(h * 960 / w)))

            show_image(frame_slot, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            render_metrics(stats)

            if is_live:
                status_slot.caption(f"LIVE | {label} | frame {stats['frame']}")
            else:
                done = min(1.0, stats["frame"] / max(1, stats["total_frames"]))
                progress.progress(done)
                status_slot.caption(
                    f"{label} | frame {stats['frame']} of {stats['total_frames']}"
                )

            # Alerts for new violations
            if stats["helmet_violations"] > state["helmet"]:
                st.toast("No-helmet violation detected", icon="🚨")

            if stats["triple_riding"] > state["triple"]:
                st.toast("Triple riding detected", icon="🚨")

            state["helmet"] = stats["helmet_violations"]
            state["triple"] = stats["triple_riding"]

            # Latest violations panel
            if stats["evidence_saved"] != state["evidence"]:
                state["evidence"] = stats["evidence_saved"]

                latest = records_to_frame(db.fetch(run_id=run_id)[:8])

                if not latest.empty:
                    show_table(
                        recent_slot,
                        latest[["bike ID", "violation", "plate"]],
                    )

        return on_frame

    if start and jobs:

        pipeline = load_pipeline(use_plates)
        last_run_id = None

        for number, (source, label, is_live) in enumerate(jobs, start=1):

            if len(jobs) > 1:
                st.toast(f"Processing {number} of {len(jobs)}: {label}")

            try:
                summary = pipeline.run(
                    source,
                    show_window=False,
                    frame_callback=make_callback(label, is_live),
                    callback_every=1 if is_photo_mode else update_every,
                    live=is_live,
                    save_video=record_video,
                )
            except RuntimeError as error:
                st.error(f"Could not open **{label}**: {error}")
                continue

            st.session_state.runs[summary["run_id"]] = {
                "label": label,
                "stats": summary,
            }
            last_run_id = summary["run_id"]

        if last_run_id:
            status_slot.success(
                "Finished. Open the **Results** tab for the full report."
            )

# -------------------- Results --------------------

with tab_results:

    runs = st.session_state.runs

    if not runs:
        st.info("Results of your runs will appear here.")
    else:
        run_ids = list(runs.keys())[::-1]  # newest first

        chosen = st.selectbox(
            "Run",
            run_ids,
            format_func=lambda r: f"{runs[r]['label']}  ({r})",
        )

        info = runs[chosen]
        stats = info["stats"]

        for col, (label, key) in zip(st.columns(len(METRICS)), METRICS):
            col.metric(label, stats.get(key, 0))

        # ----- Processed photo / video -----
        out_image = stats.get("output_image")
        out_video = stats.get("output_video")

        if out_image and os.path.exists(out_image):
            st.subheader("Processed photo")
            show_image(st, out_image)

        if out_video and os.path.exists(out_video):
            st.subheader("Processed video")

            playable = browser_video(out_video, os.path.getmtime(out_video))

            if playable:
                st.video(playable)
            else:
                st.caption(
                    "In-page playback needs ffmpeg. Download the file and open "
                    "it in any video player."
                )

            with open(out_video, "rb") as f:
                st.download_button(
                    "Download processed video", f.read(),
                    file_name=os.path.basename(out_video), mime="video/mp4",
                )

        # ----- Violations -----
        records = db.fetch(run_id=chosen)

        st.subheader("Violations")

        if not records:
            st.success("No violations were recorded in this run.")
        else:
            table = records_to_frame(records)

            show_table(
                st,
                table[["bike ID", "violation", "detected at", "time (s)",
                       "plate", "plate conf"]],
            )

            st.download_button(
                "Download violation records (CSV)",
                table.to_csv(index=False).encode("utf-8"),
                file_name=f"violations_{chosen}.csv",
                mime="text/csv",
            )

            # ----- Plates -----
            plates = [r for r in records if r["plate_number"]]
            seen_tracks = set()
            unique_plates = []

            for r in plates:
                if r["track_id"] not in seen_tracks:
                    seen_tracks.add(r["track_id"])
                    unique_plates.append(r)

            st.subheader("Number plates")

            if not unique_plates:
                st.caption("No plates were read in this run.")
            else:
                cols = st.columns(min(4, len(unique_plates)))

                for i, r in enumerate(unique_plates):
                    box = cols[i % len(cols)]

                    if r["plate_image_path"] and os.path.exists(r["plate_image_path"]):
                        show_image(box, r["plate_image_path"])

                    box.markdown(f"**{r['plate_number']}**  \nbike ID {r['track_id']}")

            # ----- Evidence -----
            st.subheader("Evidence")

            shown = [r for r in records
                     if r["evidence_path"] and os.path.exists(r["evidence_path"])]

            cols = st.columns(3)

            for i, r in enumerate(shown):
                caption = (
                    f"bike {r['track_id']} | {r['violation_type']} | "
                    f"{r['plate_number'] or 'plate not read'} | "
                    f"{r['video_time']:.1f}s"
                )
                show_image(cols[i % 3], r["evidence_path"], caption=caption)

# -------------------- History --------------------

with tab_history:

    totals = db.summary()

    c1, c2, c3 = st.columns(3)
    c1.metric("All-time helmet violations", totals["no_helmet"])
    c2.metric("All-time triple riding", totals["triple_riding"])
    c3.metric("Records with a plate", totals["with_plate"])

    history = records_to_frame(db.fetch())

    if history.empty:
        st.info("No violations logged yet.")
    else:
        f1, f2 = st.columns(2)

        kinds = sorted(history["violation"].unique())

        pick_types = f1.multiselect("Violation type", kinds, default=kinds)
        pick_runs = f2.multiselect(
            "Run", sorted(history["run_id"].unique(), reverse=True)
        )

        view = history[history["violation"].isin(pick_types)]

        if pick_runs:
            view = view[view["run_id"].isin(pick_runs)]

        show_table(
            st,
            view[["id", "run_id", "bike ID", "violation", "detected at",
                  "time (s)", "plate", "plate conf"]],
        )

        st.download_button(
            "Download all shown records (CSV)",
            view.to_csv(index=False).encode("utf-8"),
            file_name="violations.csv",
            mime="text/csv",
        )

        with st.expander("Evidence gallery (latest 30)"):
            latest = [
                r for r in db.fetch()
                if r["evidence_path"] and os.path.exists(r["evidence_path"])
            ][:30]

            cols = st.columns(3)

            for i, r in enumerate(latest):
                caption = (
                    f"bike {r['track_id']} | {r['violation_type']} | "
                    f"{r['plate_number'] or 'plate not read'}"
                )
                show_image(cols[i % 3], r["evidence_path"], caption=caption)