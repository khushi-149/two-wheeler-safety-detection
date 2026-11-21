from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime

from rider_association import (
    dedupe_motorcycles,
    dedupe_people,
    associate_riders,
    count_riders,
    detect_triple_riding,
    head_from_keypoints,
)

# -------------------- Load models --------------------

traffic_model = YOLO("yolo11n.pt")
pose_model = YOLO("yolo11n-pose.pt")
helmet_model = YOLO("models/helmet_detector.pt")

# -------------------- Output folders --------------------

evidence_folder = "outputs/evidence"
video_output_folder = "outputs/videos"

os.makedirs(evidence_folder, exist_ok=True)
os.makedirs(video_output_folder, exist_ok=True)

# -------------------- Open input video --------------------

video_path = "data/videos/no_helmet.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video!")
    raise SystemExit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_video_path = os.path.join(
    video_output_folder,
    f"processed_{timestamp}.mp4"
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (width, height)
)

if not video_writer.isOpened():
    print("Error: Could not create output video!")
    cap.release()
    raise SystemExit(1)

print("Video opened successfully!")
print("Detecting helmets, motorcycles, riders and triple riding...")
print(f"Recording processed video: {output_video_path}")
print("Press Q to close the video window.")

# -------------------- Counters --------------------

last_evidence_time = 0
evidence_count = 0
frame_count = 0
cooldown_seconds = 3

PERSON_CLASS = 0
MOTORCYCLE_CLASS = 3

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # -------------------- Traffic detection --------------------

        traffic_results = traffic_model(frame, verbose=False)[0]

        motorcycles = []

        for box in traffic_results.boxes:

            class_id = int(box.cls[0])

            if class_id == MOTORCYCLE_CLASS:

                motorcycles.append({
                    "box": box.xyxy[0].cpu().tolist(),
                    "conf": float(box.conf[0])
                })

        motorcycles = dedupe_motorcycles(motorcycles)

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
                        pose_results.keypoints.xy[index]
                        .cpu()
                        .tolist()
                    )

                    confidences = None

                    if pose_results.keypoints.conf is not None:

                        confidences = (
                            pose_results.keypoints.conf[index]
                            .cpu()
                            .tolist()
                        )

                    head = head_from_keypoints(
                        keypoints,
                        confidences
                    )

                    people.append({
                        "box": box.xyxy[0].cpu().tolist(),
                        "head": head,
                        "conf": float(box.conf[0])
                    })

        people = dedupe_people(people)

        # -------------------- Rider association --------------------

        associations = associate_riders(
            people,
            motorcycles
        )

        rider_counts = count_riders(associations)

        riding_status = detect_triple_riding(
            rider_counts
        )

        # -------------------- Helmet detection --------------------

        helmet_results = helmet_model(
            frame,
            verbose=False
        )[0]

        annotated_frame = traffic_results.plot()

        annotated_frame = helmet_results.plot(
            img=annotated_frame
        )

        no_helmet_detections = []

        for box in helmet_results.boxes:

            class_id = int(box.cls[0])

            if class_id == 1:

                no_helmet_detections.append(
                    box.xyxy[0].cpu().tolist()
                )

        no_helmet_detected = len(no_helmet_detections) > 0

        # -------------------- Draw rider counts --------------------

        triple_riding_detected = False

        for moto_index, motorcycle in enumerate(motorcycles):

            x1, y1, x2, y2 = map(
                int,
                motorcycle["box"]
            )

            count = rider_counts.get(moto_index, 0)

            status = riding_status.get(
                moto_index,
                "OK"
            )

            if status == "TRIPLE RIDING":

                color = (0, 0, 255)
                triple_riding_detected = True

            else:

                color = (0, 255, 0)

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            label = f"RIDERS: {count} | {status}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 5

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            label_y = max(
                y1 - 15,
                text_height + 20
            )

            cv2.rectangle(
                annotated_frame,
                (
                    x1,
                    label_y - text_height - 25
                ),
                (
                    x1 + text_width + 30,
                    label_y + baseline + 10
                ),
                (255, 255, 255),
                -1
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1 + 15, label_y - 5),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA
            )

        # -------------------- Draw no-helmet warnings --------------------

        for helmet_box in no_helmet_detections:

            hx1, hy1, hx2, hy2 = map(
                int,
                helmet_box
            )

            label = "WITHOUT HELMET"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.0
            thickness = 5

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            label_y = max(
                hy1 - 15,
                text_height + 20
            )

            cv2.rectangle(
                annotated_frame,
                (
                    hx1,
                    label_y - text_height - 25
                ),
                (
                    hx1 + text_width + 30,
                    label_y + baseline + 10
                ),
                (255, 255, 255),
                -1
            )

            cv2.putText(
                annotated_frame,
                label,
                (hx1 + 15, label_y - 5),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA
            )

            cv2.rectangle(
                annotated_frame,
                (hx1, hy1),
                (hx2, hy2),
                (0, 0, 255),
                3
            )

        # -------------------- Triple-riding warning --------------------

        if triple_riding_detected:

            cv2.putText(
                annotated_frame,
                "TRIPLE RIDING DETECTED",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4,
                cv2.LINE_AA
            )

        # -------------------- Save evidence --------------------

        current_time = time.time()

        if no_helmet_detected or triple_riding_detected:

            if (
                current_time - last_evidence_time
                >= cooldown_seconds
            ):

                evidence_path = os.path.join(
                    evidence_folder,
                    f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )

                if cv2.imwrite(
                    evidence_path,
                    annotated_frame
                ):

                    evidence_count += 1
                    last_evidence_time = current_time

                    print(
                        f"Evidence saved: {evidence_path}"
                    )

        # -------------------- Save processed video --------------------

        video_writer.write(annotated_frame)

        # -------------------- Display video --------------------

        display_frame = cv2.resize(
            annotated_frame,
            (960, 540)
        )

        cv2.imshow(
            "Two-Wheeler Safety Detection",
            display_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

# -------------------- Final summary --------------------

print("\nDetection completed!")
print(f"Total frames processed: {frame_count}")
print(f"Evidence screenshots saved: {evidence_count}")
print(f"Processed video saved: {output_video_path}")
