
from ultralytics import YOLO
from pathlib import Path
import torch
import csv

BASE = Path(__file__).resolve().parent

model = YOLO(str(BASE / "model" / "best.pt"))

device = 0 if torch.cuda.is_available() else "cpu"

resolutions = [640, 960, 1280]

output = BASE / "AI_resultaten" / "experiment"
output.mkdir(parents=True, exist_ok=True)

with open(output / "experiment_results.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Resolution", "Precision", "Recall",
        "F1", "mAP50", "mAP50-95",
        "Inference_ms", "FPS"
    ])

    for size in resolutions:

        print(f"\nTesting resolution: {size}")

        results = model.val(
            data=str(BASE / "data.yaml"),
            split="test",
            imgsz=size,
            batch=4,
            device=device,
            workers=0,
            conf=0.001,
            iou=0.7,
            half=False,
            plots=True,
            project=str(output),
            name=f"resolution_{size}"
        )

        precision = float(results.box.mp)
        recall = float(results.box.mr)

        f1 = (
            2 * precision * recall /
            max(precision + recall, 1e-9)
        )

        inference = results.speed["inference"]
        fps = 1000 / max(inference, 0.001)

        writer.writerow([
            size,
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(results.box.map50, 4),
            round(results.box.map, 4),
            round(inference, 2),
            round(fps, 2)
        ])

        file.flush()

print("\nExperiment voltooid.")
print(output)
