from ultralytics import YOLO
import pandas as pd
import os

# Load trained model
model = YOLO("/Users/siddhantrao/Documents/wheel-detector/runs/detect/train2/weights/best.pt")

# Path to test images
test_images_dir = "/Users/siddhantrao/Desktop/images_tested"

# Run inference
results = model.predict(source=test_images_dir, conf=0.25, save=True)
print(f"Annotated images saved in: {results[0].save_dir}")


# Prepare list for storing results
rows = []

for r in results:
    image_name = os.path.basename(r.path)
    image_width, image_height = r.orig_shape[1], r.orig_shape[0]  # width, height

    for box in r.boxes:
        # Absolute corner coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Confidence
        confidence = float(box.conf[0])

        # Class ID and name
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Normalized center coordinates and size
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        # Append row
        rows.append([
            image_name, x1, y1, x2, y2, confidence,
            class_id, class_name,
            x_center, y_center, width, height
        ])

# Create DataFrame
df = pd.DataFrame(rows, columns=[
    "image_name", "x1", "y1", "x2", "y2", "confidence",
    "class_id", "class_name",
    "x_center", "y_center", "width", "height"
])

# Save to CSV on Desktop
csv_path = "/Users/siddhantrao/Desktop/yolo_predictions_full.csv"
df.to_csv(csv_path, index=False)

print(f"✅ Predictions CSV saved at: {csv_path}")
