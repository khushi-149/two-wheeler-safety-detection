# Data

## Folders

| Folder | Content |
|---|---|
| `data/videos/` | Traffic videos to analyse (`.mp4 .avi .mov .mkv .m4v .webm`); uploads from the dashboard go to `data/videos/uploads/` |
| `data/images/` | Photos to analyse in the dashboard; uploads go to `data/images/uploads/` |
| `data/test_images/` | Photos for `src/interactive_tester.py` |
| `data/helmet_dataset/` | Training dataset (only needed to retrain the model) |

Videos, photos and the dataset are not stored in git.

## Helmet detection dataset

- Name: Helmet Detection_YOLOv8 - With and Without Helmet
- Source: Roboflow Universe
- Images: 4,169
- License: CC BY 4.0 (attribution required)
- Dataset: https://universe.roboflow.com/learning-evidence/helmet-detection_yolov8

Classes: `With Helmet`, `Without Helmet`.

Download it in **YOLOv8** format and extract it to `data/helmet_dataset/` so that
`data/helmet_dataset/data.yaml` exists (paths inside must point at the extracted train / valid /
test image folders).