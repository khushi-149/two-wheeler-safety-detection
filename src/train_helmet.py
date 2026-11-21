
from ultralytics import YOLO

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train the model on our helmet dataset
model.train(
    data="data/helmet_dataset/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    workers=0,
    project="runs/helmet",
    name="helmet_detector"
)

print("Helmet model training completed!")
