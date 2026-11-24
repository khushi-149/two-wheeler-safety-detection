# Model weights

Weight files are not stored in git (`*.pt` is ignored). Place them in this folder.

| File | Purpose | Where it comes from |
|---|---|---|
| `helmet_detector.pt` | **Required.** Custom YOLO11n, classes *With Helmet* / *Without Helmet* | Download from this repository's **Releases** page, or train it (`src/train_helmet.py`, `notebooks/`) |
| `yolo11n.pt` | Motorcycle detection and tracking | Downloaded automatically by Ultralytics on first use |
| `yolo11n-pose.pt` | Rider pose / hip and head keypoints | Downloaded automatically on first use |
| `plate_detector.pt` | *Optional.* Number-plate detector (one class); without it plate reading stays off | Train it with `notebooks/plate_detection_training.ipynb` on Google Colab and save it with this name |

Check what is in place with `python scripts/check_setup.py`.

The application reads the helmet model's class names, so a retrained model with the classes in
a different order still works as long as the names contain "helmet" / "without".