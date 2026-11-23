import cv2
from ultralytics import YOLO

from rider_tracking import MotorcycleTracker

VIDEO_PATH = "data/videos/no_helmet.mp4"

model = YOLO("yolo11n.pt")
tracker = MotorcycleTracker(model)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit(f"Could not open video: {VIDEO_PATH}")

frame_count = 0
tracked_frames = 0
seen_ids = set()

print("Testing ByteTrack on actual video...")

try:
    while frame_count < 300:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        motorcycles = tracker.update(frame)

        if motorcycles:
            tracked_frames += 1

        for motorcycle in motorcycles:
            track_id = motorcycle["track_id"]
            seen_ids.add(track_id)

        if frame_count % 30 == 0:
            print(
                f"Frame {frame_count}: "
                f"{len(motorcycles)} motorcycles tracked | "
                f"IDs: {[m['track_id'] for m in motorcycles]}"
            )

finally:
    cap.release()

print("\nTracking test completed!")
print("Frames processed:", frame_count)
print("Frames with motorcycles:", tracked_frames)
print("Unique tracking IDs:", sorted(seen_ids))
