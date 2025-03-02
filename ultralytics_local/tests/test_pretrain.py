from ultralytics_local.ultralytics import YOLO
from ultralytics_local.ultralytics.models.yolo import obb

model = YOLO('yolo11n-obb.pt')
print(model.cfg)
for k, v in model.ckpt.get("train_results", {}).items():
    print(f"{k}: {v}")
