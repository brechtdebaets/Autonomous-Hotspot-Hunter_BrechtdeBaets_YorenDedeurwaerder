
from pathlib import Path
from datetime import datetime
from tkinter import Tk, filedialog
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import cv2
import torch
import csv

BASE = Path(__file__).resolve().parent
DATA = BASE / "thermal_images"
MODEL = BASE / "model" / "best.pt"
OUTPUT = BASE / "AI_resultaten" / "thermal"

OUTPUT.mkdir(parents=True, exist_ok=True)

CONFIDENCE = 0.15
RESOLUTION = 640
TEMP_CONTRAST = 15
MIN_AREA = 12

device = 0 if torch.cuda.is_available() else "cpu"


def read_thermal(source):

    with Image.open(source) as image:
        print("TIFF mode:", image.mode)
        thermal = np.asarray(image)

    thermal = np.squeeze(thermal)

    print("TIFF shape:", thermal.shape)
    print("TIFF datatype:", thermal.dtype)

    if thermal.ndim == 3:

        if np.allclose(
            thermal,
            thermal[:, :, :1],
            equal_nan=True
        ):
            thermal = thermal[:, :, 0]

    if thermal.ndim != 2:
        raise ValueError(
            f"Unsupported TIFF structure: {thermal.shape}"
        )

    if not np.issubdtype(thermal.dtype, np.floating):
        raise ValueError(
            f"Unexpected temperature format: {thermal.dtype}"
        )

    return thermal.astype(np.float32)


def find_rgb(source):

    category = source.parent.parent.parent
    rgb_folder = category / "RGB" / "Corrected FOV"

    if not rgb_folder.exists():
        return None

    for file in rgb_folder.iterdir():

        if (
            file.stem == source.stem
            and file.suffix.lower() in
            [".jpg", ".jpeg", ".png"]
        ):
            return file

    return None


def analyse(source, model):

    source = Path(source)
    category = source.parent.parent.parent.name

    print("\n========================================")
    print("Image:", source.name)
    print("Category:", category)

    rgb_path = find_rgb(source)

    if rgb_path is None:
        print("Matching RGB image not found.")
        return None

    rgb = cv2.imread(str(rgb_path))

    if rgb is None:
        print("Cannot open RGB image.")
        return None

    thermal = read_thermal(source)

    valid = np.isfinite(thermal)

    if not valid.any():
        print("No valid temperature values.")
        return None

    background = float(np.median(thermal[valid]))
    minimum = float(np.min(thermal[valid]))
    maximum = float(np.max(thermal[valid]))
    average = float(np.mean(thermal[valid]))

    threshold = background + TEMP_CONTRAST

    mask = (
        valid & (thermal >= threshold)
    ).astype(np.uint8)

    count, labels, stats, centers = (
        cv2.connectedComponentsWithStats(mask)
    )

    prediction = model.predict(
        rgb,
        conf=CONFIDENCE,
        imgsz=RESOLUTION,
        device=device,
        verbose=False
    )[0]

    annotated = prediction.plot()

    fire_count = 0
    smoke_count = 0

    for box in prediction.boxes:

        label = model.names[int(box.cls.item())]

        if label == "fire":
            fire_count += 1

        elif label == "smoke":
            smoke_count += 1

    rows = []

    for index in range(1, count):

        x, y, w, h, area = stats[index]

        if area < MIN_AREA:
            continue

        region = thermal[labels == index]

        peak = float(np.max(region))
        mean = float(np.mean(region))

        cx, cy = centers[index]

        hotspot_id = len(rows) + 1

        rows.append([
            hotspot_id,
            round(float(cx), 2),
            round(float(cy), 2),
            int(area),
            round(peak, 2),
            round(mean, 2),
            round(peak - background, 2),
            int(x),
            int(y),
            int(w),
            int(h)
        ])

    low = float(np.percentile(thermal[valid], 2))
    high = maximum

    if high <= low:
        high = low + 1

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        facecolor="#15191e"
    )

    axes[0, 0].imshow(
        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    )

    axes[0, 0].set_title(
        "Original RGB image",
        color="white"
    )

    axes[0, 1].imshow(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    )

    axes[0, 1].set_title(
        f"YOLO11n | Fire: {fire_count} | Smoke: {smoke_count}",
        color="white"
    )

    thermal_plot = axes[1, 0].imshow(
        thermal,
        cmap="inferno",
        vmin=low,
        vmax=high
    )

    axes[1, 0].set_title(
        "Radiometric thermal image",
        color="white"
    )

    axes[1, 1].imshow(
        thermal,
        cmap="inferno",
        vmin=low,
        vmax=high
    )

    axes[1, 1].set_title(
        f"Thermal inspection candidates: {len(rows)}",
        color="white"
    )

    for row in rows:

        hotspot_id = row[0]
        peak = row[4]

        x, y, w, h = row[7:11]

        rectangle = Rectangle(
            (x, y),
            w,
            h,
            linewidth=1.5,
            edgecolor="cyan",
            facecolor="none"
        )

        axes[1, 1].add_patch(rectangle)

        axes[1, 1].text(
            x,
            max(5, y - 5),
            f"H{hotspot_id}: {peak:.1f} C",
            color="white",
            fontsize=8,
            bbox=dict(
                facecolor="black",
                alpha=0.7,
                edgecolor="none"
            )
        )

    for ax in axes.flat:
        ax.axis("off")

    fig.subplots_adjust(
        left=0.03,
        right=0.91,
        top=0.90,
        bottom=0.12,
        wspace=0.08,
        hspace=0.20
    )

    color_axis = fig.add_axes(
        [0.93, 0.15, 0.015, 0.32]
    )

    colorbar = fig.colorbar(
        thermal_plot,
        cax=color_axis
    )

    colorbar.set_label(
        "Temperature (C)",
        color="white"
    )

    colorbar.ax.tick_params(colors="white")

    fig.suptitle(
        f"HOTSPOT HUNTER | {category} | {source.stem}",
        color="white",
        fontsize=17,
        fontweight="bold"
    )

    fig.text(
        0.5,
        0.065,
        f"Background: {background:.1f} C   |   "
        f"Maximum: {maximum:.1f} C   |   "
        f"Threshold: {threshold:.1f} C",
        color="white",
        fontsize=12,
        ha="center"
    )

    fig.text(
        0.5,
        0.035,
        f"Thermal candidates: {len(rows)}   |   "
        "UGV verification required (simulation)",
        color="cyan",
        fontsize=11,
        ha="center"
    )

    folder = OUTPUT / (
        category.replace(" ", "_") + "_" +
        source.stem + "_" +
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    folder.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        folder / "comparison.png",
        dpi=160,
        facecolor=fig.get_facecolor()
    )

    cv2.imwrite(
        str(folder / "thermal_mask.png"),
        mask * 255
    )

    cv2.imwrite(
        str(folder / "yolo_result.jpg"),
        annotated
    )

    with open(
        folder / "hotspots.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Hotspot_ID",
            "Pixel_X",
            "Pixel_Y",
            "Area_pixels",
            "Maximum_C",
            "Mean_C",
            "Contrast_C",
            "X",
            "Y",
            "Width",
            "Height"
        ])

        writer.writerows(rows)

    summary = [
        source.name,
        category,
        round(minimum, 2),
        round(average, 2),
        round(background, 2),
        round(maximum, 2),
        round(threshold, 2),
        fire_count,
        smoke_count,
        len(rows)
    ]

    headers = [
        "Image",
        "Category",
        "Minimum_C",
        "Average_C",
        "Background_C",
        "Maximum_C",
        "Threshold_C",
        "Fire_detections",
        "Smoke_detections",
        "Thermal_candidates"
    ]

    with open(
        folder / "summary.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerow(summary)

    print(f"Minimum temperature: {minimum:.1f} C")
    print(f"Average temperature: {average:.1f} C")
    print(f"Background temperature: {background:.1f} C")
    print(f"Maximum temperature: {maximum:.1f} C")
    print(f"Thermal threshold: {threshold:.1f} C")
    print(f"Fire detections: {fire_count}")
    print(f"Smoke detections: {smoke_count}")
    print(f"Thermal candidates: {len(rows)}")
    print(f"Results: {folder}")

    plt.show()
    plt.close(fig)

    return summary


root = Tk()
root.withdraw()
root.attributes("-topmost", True)

sources = filedialog.askopenfilenames(
    initialdir=str(DATA),
    title="Select Celsius TIFF images",
    filetypes=[("Thermal TIFF", "*.tif *.tiff")]
)

root.destroy()

if not sources:
    raise SystemExit

model = YOLO(str(MODEL))

summary_path = OUTPUT / (
    "thermal_comparison_" +
    datetime.now().strftime("%Y%m%d_%H%M%S") +
    ".csv"
)

with open(summary_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Image",
        "Category",
        "Minimum_C",
        "Average_C",
        "Background_C",
        "Maximum_C",
        "Threshold_C",
        "Fire_detections",
        "Smoke_detections",
        "Thermal_candidates"
    ])

    for source in sources:

        try:

            result = analyse(source, model)

            if result is not None:
                writer.writerow(result)
                file.flush()

        except Exception as error:

            print("Analysis failed:", source)
            print("Error:", error)

print("\nThermal analysis finished.")
print("Comparison:", summary_path)
