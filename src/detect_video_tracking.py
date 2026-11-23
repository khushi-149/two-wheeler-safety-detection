import cv2
import os
from datetime import datetime

from ultralytics import YOLO

from rider_association import (
    dedupe_motorcycles,
    dedupe_people,
    associate_riders,
    count_riders,
    detect_triple_riding,
    head_from_keypoints,
)

from rider_tracking import MotorcycleTracker


# -------------------- Settings --------------------

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Detection thresholds
MOTORCYCLE_CONF = 0.25
HELMET_CONF = 0.40

# Evidence: minimum video-seconds before the same violation on the
# same motorcycle track is saved again.
COOLDOWN_SECONDS = 3

PERSON_CLASS = 0
HELMET_CLASS = 0       # "with helmet"  (check your model's class order)
NO_HELMET_CLASS = 1    # "without helmet"

# How far the motorcycle box is expanded when checking whether a
# no-helmet head belongs to it. Riders' heads sit ABOVE the bike box.
EXPAND_TOP = 1.0       # 100% of box height upward
EXPAND_SIDE = 0.15     # 15% of box width on each side
EXPAND_BOTTOM = 0.0

# Colors (BGR)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)

# These are set after the video is opened (resolution-aware)
LABEL_SCALE = 0.85
LABEL_THICKNESS = 2
WARNING_SCALE = 1.15
WARNING_THICKNESS = 3
BOX_THICKNESS = 3
THIN_THICKNESS = 2
LABEL_PAD = 6


# -------------------- Helper functions --------------------

def draw_label(frame, text, x, y, color=WHITE,
               font_scale=None, thickness=None, bg=DARK):
    """Draw text on a dark background, clamped inside the frame."""

    font_scale = font_scale or LABEL_SCALE
    thickness = thickness or LABEL_THICKNESS

    (text_width, text_height), baseline = cv2.getTextSize(
        text, FONT, font_scale, thickness
    )

    pad = LABEL_PAD

    x = int(max(0, min(x, frame.shape[1] - text_width - 2 * pad)))
    y = int(max(text_height + 2 * pad, y))

    cv2.rectangle(
        frame,
        (x, y - text_height - pad),
        (x + text_width + 2 * pad, y + baseline + pad // 2),
        bg,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + pad, y),
        FONT,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_warning(frame, text, y):
    draw_label(
        frame,
        text,
        20,
        y,
        WHITE,
        font_scale=WARNING_SCALE,
        thickness=WARNING_THICKNESS,
        bg=(0, 0, 200),
    )


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def expanded_box(box):
    """Expand a motorcycle box upward/sideways to include riders' heads."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    return (
        x1 - EXPAND_SIDE * w,
        y1 - EXPAND_TOP * h,
        x2 + EXPAND_SIDE * w,
        y2 + EXPAND_BOTTOM * h,
    )


def point_in_box(point, box):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def match_no_helmet_to_motorcycle(no_helmet_box, motorcycles):
    """
    Return the index of the motorcycle this no-helmet head belongs to,
    or None (e.g. a pedestrian). If several bikes overlap, the one whose
    box center is closest to the head wins.
    """
    center = box_center(no_helmet_box)
    best_index = None
    best_distance = float("inf")

    for index, moto in enumerate(motorcycles):

        if not point_in_box(center, expanded_box(moto["box"])):
            continue

        mx, my = box_center(moto["box"])
        distance = (mx - center[0]) ** 2 + (my - center[1]) ** 2

        if distance < best_distance:
            best_distance = distance
            best_index = index

    return best_index


# -------------------- Load models --------------------

traffic_model = YOLO("yolo11n.pt")
pose_model = YOLO("yolo11n-pose.pt")
helmet_model = YOLO("models/helmet_detector.pt")

motorcycle_tracker = MotorcycleTracker(
    traffic_model,
    confidence=MOTORCYCLE_CONF,
)


# -------------------- Output folders --------------------

evidence_folder = "outputs/evidence"
video_output_folder = "outputs/videos"

os.makedirs(evidence_folder, exist_ok=True)
os.makedirs(video_output_folder, exist_ok=True)


# -------------------- Open input video --------------------

video_path = "data/videos/no_helmet.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise SystemExit("Error: Could not open video!")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

if width <= 0 or height <= 0:
    cap.release()
    raise SystemExit("Error: Invalid video dimensions!")


# -------------------- Resolution-aware annotation sizes --------------------

scale_factor = max(0.6, height / 720)

LABEL_SCALE = 0.7 * scale_factor
LABEL_THICKNESS = max(2, int(2 * scale_factor))
WARNING_SCALE = 1.0 * scale_factor
WARNING_THICKNESS = max(2, int(3 * scale_factor))
BOX_THICKNESS = max(2, int(3 * scale_factor))
THIN_THICKNESS = max(1, int(2 * scale_factor))
LABEL_PAD = max(6, int(6 * scale_factor))
WARNING_LINE_HEIGHT = int(65 * scale_factor)
WARNING_START_Y = int(55 * scale_factor)


# -------------------- Output video --------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_video_path = os.path.join(
    video_output_folder,
    f"tracked_{timestamp}.mp4",
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (width, height),
)

if not video_writer.isOpened():
    cap.release()
    raise SystemExit("Error: Could not create output video!")


print("Video opened successfully!")
print("Running detection with motorcycle tracking...")
print(f"Original video resolution: {width}x{height}")
print(f"FPS: {fps:.2f}")
print(f"Output video: {output_video_path}")
print("Press Q to close the video window.")


# -------------------- Counters / state --------------------

frame_count = 0
evidence_count = 0

# (track_id, violation_type) -> video time of last saved evidence
last_evidence_time = {}

# Unique violators over the whole clip (useful for the later database phase)
helmet_violator_ids = set()
triple_riding_ids = set()


try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        video_time = frame_count / fps

        # -------------------- Motorcycle tracking --------------------

        tracked_motorcycles = motorcycle_tracker.update(frame)

        motorcycles = [
            {
                "box": moto["box"],
                "conf": moto["conf"],
                "track_id": moto["track_id"],
            }
            for moto in tracked_motorcycles
        ]

        # -------------------- Pose detection --------------------

        people = []

        if motorcycles:

            pose_results = pose_model(frame, verbose=False)[0]

            if (
                pose_results.boxes is not None
                and pose_results.keypoints is not None
            ):

                for index, box in enumerate(pose_results.boxes):

                    class_id = int(box.cls[0])

                    if class_id != PERSON_CLASS:
                        continue

                    keypoints = (
                        pose_results.keypoints.xy[index].cpu().tolist()
                    )

                    confidences = None

                    if pose_results.keypoints.conf is not None:
                        confidences = (
                            pose_results.keypoints.conf[index]
                            .cpu()
                            .tolist()
                        )

                    head = head_from_keypoints(keypoints, confidences)

                    people.append({
                        "box": box.xyxy[0].cpu().tolist(),
                        "head": head,
                        "conf": float(box.conf[0]),
                    })

        people = dedupe_people(people)

        # -------------------- Rider association --------------------

        association_motorcycles = [
            {
                "box": moto["box"],
                "conf": moto["conf"],
            }
            for moto in motorcycles
        ]

        associations = associate_riders(people, association_motorcycles)

        rider_counts = count_riders(associations)

        riding_status = detect_triple_riding(rider_counts)

        # -------------------- Helmet detection --------------------

        helmet_results = helmet_model(
            frame,
            conf=HELMET_CONF,
            verbose=False,
        )[0]

        # Start from the clean frame; we draw everything ourselves.
        annotated_frame = frame.copy()

        # motorcycle index -> list of no-helmet boxes
        no_helmet_by_moto = {}

        for box in helmet_results.boxes:

            class_id = int(box.cls[0])
            coords = box.xyxy[0].cpu().tolist()

            hx1, hy1, hx2, hy2 = map(int, coords)

            if class_id == NO_HELMET_CLASS:

                moto_index = match_no_helmet_to_motorcycle(
                    coords,
                    motorcycles,
                )

                # Ignore no-helmet heads that don't belong to any
                # tracked motorcycle (pedestrians, parked people, etc.)
                if moto_index is None:
                    continue

                no_helmet_by_moto.setdefault(moto_index, []).append(coords)

                cv2.rectangle(
                    annotated_frame,
                    (hx1, hy1),
                    (hx2, hy2),
                    RED,
                    BOX_THICKNESS,
                )

                draw_label(
                    annotated_frame,
                    "WITHOUT HELMET",
                    hx1,
                    hy1 - 8,
                    WHITE,
                    bg=(0, 0, 180),
                )

            elif class_id == HELMET_CLASS:

                # Thin green box, no label, for helmeted riders
                cv2.rectangle(
                    annotated_frame,
                    (hx1, hy1),
                    (hx2, hy2),
                    (0, 200, 0),
                    THIN_THICKNESS,
                )

        no_helmet_detected = len(no_helmet_by_moto) > 0

        # -------------------- Draw tracked motorcycles --------------------

        triple_riding_detected = False

        # Violations found this frame: list of (track_id, type)
        frame_violations = []

        for moto_index, motorcycle in enumerate(motorcycles):

            x1, y1, x2, y2 = map(int, motorcycle["box"])

            track_id = motorcycle["track_id"]

            count = rider_counts.get(moto_index, 0)
            status = riding_status.get(moto_index, "OK")

            has_no_helmet = moto_index in no_helmet_by_moto
            is_triple = status == "TRIPLE RIDING"

            if is_triple:
                triple_riding_detected = True
                triple_riding_ids.add(track_id)
                frame_violations.append((track_id, "triple_riding"))

            if has_no_helmet:
                helmet_violator_ids.add(track_id)
                frame_violations.append((track_id, "no_helmet"))

            color = RED if (is_triple or has_no_helmet) else GREEN

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                BOX_THICKNESS,
            )

            label = f"ID: {track_id} | RIDERS: {count} | {status}"

            if has_no_helmet:
                label += " | NO HELMET"

            draw_label(
                annotated_frame,
                label,
                x1,
                y1 - 8,
                WHITE,
            )

        # -------------------- Violation warnings --------------------

        warning_y = WARNING_START_Y

        if no_helmet_detected:

            draw_warning(annotated_frame, "NO HELMET VIOLATION", warning_y)
            warning_y += WARNING_LINE_HEIGHT

        if triple_riding_detected:

            draw_warning(annotated_frame, "TRIPLE RIDING DETECTED", warning_y)

        # -------------------- Save evidence --------------------
        # One screenshot per (motorcycle track, violation type), repeated
        # only after COOLDOWN_SECONDS of *video* time.

        save_frame = False

        for key in frame_violations:

            last_time = last_evidence_time.get(key)

            if last_time is None or video_time - last_time >= COOLDOWN_SECONDS:
                save_frame = True
                last_evidence_time[key] = video_time

        if save_frame:

            evidence_path = os.path.join(
                evidence_folder,
                f"tracked_violation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg",
            )

            if cv2.imwrite(evidence_path, annotated_frame):

                evidence_count += 1

                kinds = ", ".join(
                    f"ID {tid}: {vtype}" for tid, vtype in frame_violations
                )

                print(
                    f"Evidence saved ({video_time:.1f}s) [{kinds}]: "
                    f"{evidence_path}"
                )

        # -------------------- Save processed video --------------------

        video_writer.write(annotated_frame)

        # -------------------- Display video (keep aspect ratio) --------------------

        display_scale = min(DISPLAY_WIDTH / width, DISPLAY_HEIGHT / height)

        display_frame = cv2.resize(
            annotated_frame,
            (int(width * display_scale), int(height * display_scale)),
        )

        cv2.imshow("Two-Wheeler Safety - ByteTrack", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    video_writer.release()

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


# -------------------- Final summary --------------------

print("\nTracking and detection completed!")
print(f"Total frames processed: {frame_count}")
print(f"Evidence screenshots saved: {evidence_count}")
print(f"Motorcycles with helmet violations: {sorted(helmet_violator_ids)}")
print(f"Motorcycles with triple riding: {sorted(triple_riding_ids)}")
print(f"Processed video saved: {output_video_path}")