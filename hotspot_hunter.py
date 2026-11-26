
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from tkinter import Tk, filedialog, simpledialog
import cv2
import torch
import numpy as np
import csv
import time
import sys

BASE = Path(__file__).resolve().parent
MODEL = BASE / "model" / "best.pt"
VIDEOS = BASE / "demo_video"
OUTPUT = BASE / "AI_resultaten"

device = 0 if torch.cuda.is_available() else "cpu"
extensions = [".mp4", ".avi", ".mov", ".mkv", ".jpg", ".png", ".jpeg"]

OUTPUT.mkdir(exist_ok=True)


def analyse(source, confidence=0.25, resolution=640, show=True):

    source = Path(source)
    model = YOLO(str(MODEL))

    output = OUTPUT / (
        source.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=True)

    is_image = source.suffix.lower() in [".jpg", ".jpeg", ".png"]
    cap = None if is_image else cv2.VideoCapture(str(source))

    if cap is not None and not cap.isOpened():
        raise RuntimeError("Video kan niet geopend worden.")

    source_fps = cap.get(cv2.CAP_PROP_FPS) if cap else 1

    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = 25

    tracked_ids = set()
    fire_tracks = {}
    inspection_queue = set()

    frame_number = 0
    total_fire = 0
    total_smoke = 0
    writer = None

    start_total = time.perf_counter()

    print(f"\nAnalyse: {source.name}")

    try:
        with open(output / "detections.csv", "w", newline="") as file:

            log = csv.writer(file)

            log.writerow([
                "frame", "time_s", "track_id", "class",
                "confidence", "x1", "y1", "x2", "y2"
            ])

            while True:

                if is_image:
                    frame = cv2.imread(str(source))
                    ret = frame is not None
                else:
                    ret, frame = cap.read()

                if not ret:
                    break

                start = time.perf_counter()

                settings = {
                    "conf": confidence,
                    "imgsz": resolution,
                    "device": device,
                    "verbose": False
                }

                if is_image:
                    result = model.predict(frame, **settings)[0]
                else:
                    result = model.track(
                        frame,
                        persist=True,
                        tracker="botsort.yaml",
                        **settings
                    )[0]

                fire_count = 0
                smoke_count = 0

                for box in result.boxes:

                    label = model.names[int(box.cls.item())]
                    score = float(box.conf.item())

                    track_id = (
                        int(box.id.item())
                        if box.id is not None else None
                    )

                    coords = box.xyxy[0].tolist()

                    if label == "fire":
                        fire_count += 1
                        total_fire += 1

                    elif label == "smoke":
                        smoke_count += 1
                        total_smoke += 1

                    if track_id is not None:

                        tracked_ids.add(track_id)

                        if label == "fire":

                            fire_tracks[track_id] = (
                                fire_tracks.get(track_id, 0) + 1
                            )

                            if fire_tracks[track_id] >= 10:
                                inspection_queue.add(track_id)

                    log.writerow([
                        frame_number,
                        round(frame_number / source_fps, 2),
                        track_id,
                        label,
                        round(score, 4),
                        *coords
                    ])

                fps = 1 / max(time.perf_counter() - start, 0.001)

                annotated = result.plot()

                scale = min(
                    720 / annotated.shape[0],
                    1280 / annotated.shape[1]
                )

                width = int(annotated.shape[1] * scale)
                height = int(annotated.shape[0] * scale)

                width = max(2, width - width % 2)
                height = max(2, height - height % 2)

                annotated = cv2.resize(
                    annotated, (width, height)
                )

                panel = np.full(
                    (height, 360, 3),
                    (24, 28, 32),
                    dtype=np.uint8
                )

                cv2.putText(
                    panel, "HOTSPOT HUNTER",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2
                )

                lines = [
                    "UAV MONITORING",
                    f"Fire detections: {fire_count}",
                    f"Smoke detections: {smoke_count}",
                    f"Unique tracks: {len(tracked_ids)}",
                    f"Analysis FPS: {fps:.1f}",
                    f"Confidence: {confidence:.2f}",
                    f"Resolution: {resolution}",
                    "",
                    "UGV SIMULATION",
                    f"Candidates: {len(inspection_queue)}",
                    "Minimum observations: 10",
                    "Awaiting operator",
                    "",
                    "Q - Stop analysis"
                ]

                spacing = min(27, (height - 70) / len(lines))

                for i, text in enumerate(lines):

                    color = (210, 215, 220)

                    if text in ["UAV MONITORING", "UGV SIMULATION"]:
                        color = (80, 180, 255)

                    cv2.putText(
                        panel, text,
                        (20, int(65 + i * spacing)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1
                    )

                display = np.hstack((annotated, panel))

                frame_number += 1

                if is_image:

                    cv2.imwrite(
                        str(output / "result.jpg"), display
                    )

                    if show:
                        cv2.imshow("Hotspot Hunter", display)
                        cv2.waitKey(0)

                    break

                if writer is None:

                    h, w = display.shape[:2]

                    writer = cv2.VideoWriter(
                        str(output / "result.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (w, h)
                    )

                    if not writer.isOpened():
                        raise RuntimeError("Video-export mislukt.")

                writer.write(display)

                if show:

                    cv2.imshow("Hotspot Hunter", display)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    finally:

        if cap is not None:
            cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    duration = time.perf_counter() - start_total

    summary = [
        source.name,
        frame_number,
        total_fire,
        total_smoke,
        len(tracked_ids),
        len(inspection_queue),
        round(duration, 2)
    ]

    with open(output / "summary.csv", "w", newline="") as file:

        log = csv.writer(file)

        log.writerow([
            "Video", "Frames", "Fire", "Smoke",
            "Tracks", "UGV_candidates", "Processing_time_s"
        ])

        log.writerow(summary)

    print("Analyse voltooid.")
    print(f"Frames: {frame_number}")
    print(f"Vuurdetecties: {total_fire}")
    print(f"Rookdetecties: {total_smoke}")
    print(f"Unieke tracks: {len(tracked_ids)}")
    print(f"Inspectiekandidaten: {len(inspection_queue)}")
    print(f"Resultaten: {output}")

    return summary


def analyse_all():

    files = sorted(
        file for file in VIDEOS.iterdir()
        if file.suffix.lower() in extensions
    )

    print(f"{len(files)} bestanden gevonden.")

    summary_path = OUTPUT / "video_comparison.csv"

    with open(summary_path, "w", newline="") as file:

        log = csv.writer(file)

        log.writerow([
            "Video", "Frames", "Fire", "Smoke",
            "Tracks", "UGV_candidates", "Processing_time_s"
        ])

        for number, source in enumerate(files, 1):

            print(f"\nBestand {number}/{len(files)}")

            try:
                summary = analyse(
                    source,
                    confidence=0.15,
                    resolution=640,
                    show=True
                )

                log.writerow(summary)
                file.flush()

            except Exception as error:
                print(f"Fout bij {source.name}: {error}")

    print("\nAlle analyses afgewerkt.")
    print(summary_path)


def select_video():

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    source = filedialog.askopenfilename(
        initialdir=str(VIDEOS),
        title="Selecteer beeldmateriaal",
        filetypes=[
            ("Video en afbeeldingen",
             "*.mp4 *.avi *.mov *.mkv *.jpg *.png *.jpeg")
        ]
    )

    if not source:
        root.destroy()
        return

    confidence = simpledialog.askfloat(
        "Confidence",
        "Confidence threshold:",
        initialvalue=0.25,
        minvalue=0.01,
        maxvalue=1.0
    )

    resolution = simpledialog.askinteger(
        "Resolutie",
        "Invoerresolutie:",
        initialvalue=640,
        minvalue=320,
        maxvalue=1280
    )

    root.destroy()

    if confidence is not None and resolution is not None:
        analyse(source, confidence, resolution)


if __name__ == "__main__":

    if "--all" in sys.argv:
        analyse_all()
    else:
        select_video()
