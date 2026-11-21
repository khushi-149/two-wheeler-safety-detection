from ultralytics import YOLO
import cv2

# Load pretrained YOLO model for traffic detection
traffic_model = YOLO("yolo11n.pt")

# Load custom helmet detection model
helmet_model = YOLO("models/helmet_detector.pt")

# Open traffic video
video_path = "data/videos/traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

print("Video opened successfully!")
print("Detecting traffic and helmet violations...")
print("Press Q to close the video window.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect traffic objects
    traffic_results = traffic_model(frame, verbose=False)

    # Detect helmets
    helmet_results = helmet_model(frame, verbose=False)

    # Draw traffic detection boxes
    annotated_frame = traffic_results[0].plot()

    # Draw helmet detection boxes
    annotated_frame = helmet_results[0].plot(
        img=annotated_frame
    )

    # Resize video for comfortable screen viewing
    display_frame = cv2.resize(annotated_frame, (960, 540))

    # Display video
    cv2.imshow("Two-Wheeler Safety Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Detection completed!")
