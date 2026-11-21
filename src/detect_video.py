from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime

# Load models
traffic_model = YOLO("yolo11n.pt")
helmet_model = YOLO("models/helmet_detector.pt")

# Create output folders
evidence_folder = "outputs/evidence"
video_output_folder = "outputs/videos"

os.makedirs(evidence_folder, exist_ok=True)
os.makedirs(video_output_folder, exist_ok=True)

# Open input video
video_path = "data/videos/no_helmet.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

# Create output video
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
    exit()

print("Video opened successfully!")
print("Detecting traffic and no-helmet cases...")
print(f"Recording processed video: {output_video_path}")
print("Press Q to close the video window.")

last_evidence_time = 0
evidence_count = 0
frame_count = 0
cooldown_seconds = 3

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Detect traffic objects
    traffic_results = traffic_model(frame, verbose=False)

    # Detect helmet status
    helmet_results = helmet_model(frame, verbose=False)

    # Draw detection boxes
    annotated_frame = traffic_results[0].plot()
    annotated_frame = helmet_results[0].plot(img=annotated_frame)

    # Check for Without Helmet detections (class ID 1)
    no_helmet_detected = False

    for box in helmet_results[0].boxes:
        class_id = int(box.cls[0])

        if class_id == 1:
            no_helmet_detected = True
            break

    # Save evidence screenshots
    current_time = time.time()

    if no_helmet_detected:
        cv2.putText(
            annotated_frame,
            "WITHOUT HELMET DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if current_time - last_evidence_time >= cooldown_seconds:
            evidence_path = os.path.join(
                evidence_folder,
                f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            )

            cv2.imwrite(evidence_path, annotated_frame)
            evidence_count += 1
            last_evidence_time = current_time

            print(f"Evidence saved: {evidence_path}")

    # Write annotated frame to output video
    video_writer.write(annotated_frame)

    # Resize video for comfortable screen viewing
    display_frame = cv2.resize(annotated_frame, (960, 540))

    # Display video
    cv2.imshow("Two-Wheeler Safety Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("\nDetection completed!")
print(f"Total frames processed: {frame_count}")
print(f"Evidence screenshots saved: {evidence_count}")
print(f"Processed video saved: {output_video_path}")
