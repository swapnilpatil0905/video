import csv
import os
import cv2

INPUT_DIR = "scenes/frames"
MODEL = "models/facedetection/face_detection_yunet_2023mar.onnx"

OUTPUT_DIR = "faces/crops"
OUTPUT_CSV = "faces/face_manifest.csv"

CONFIDENCE_THRESHOLD = 0.75
PADDING = 0.35
CROP_SIZE = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL):
    raise FileNotFoundError(f"Face detection model not found: {MODEL}")

detector = cv2.FaceDetectorYN_create(
    MODEL,
    "",
    (320, 320),
    CONFIDENCE_THRESHOLD,
    0.3,
    5000
)

files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

manifest = []
face_id = 0

for index, filename in enumerate(files, 1):

    path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(path)

    if image is None:
        print(f"Could not read: {filename}")
        continue

    height, width = image.shape[:2]

    detector.setInputSize((width, height))
    _, faces = detector.detect(image)

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4]
            confidence = float(face[-1])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Add padding around the detected face.
            cx = x + w / 2
            cy = y + h / 2

            side = max(w, h) * (1 + 2 * PADDING)

            x1 = max(0, int(cx - side / 2))
            y1 = max(0, int(cy - side / 2))
            x2 = min(width, int(cx + side / 2))
            y2 = min(height, int(cy + side / 2))

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))

            face_id += 1
            crop_filename = f"face_{face_id:06d}.jpg"
            crop_path = os.path.join(OUTPUT_DIR, crop_filename)

            cv2.imwrite(
                crop_path,
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

            manifest.append({
                "face_id": face_id,
                "scene": filename,
                "face_crop": crop_path,
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "width": round(float(w), 2),
                "height": round(float(h), 2),
                "confidence": round(confidence, 4)
            })

    if index % 100 == 0:
        print(
            f"Processed {index}/{len(files)} scenes | "
            f"Faces detected: {face_id}"
        )

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "face_id",
            "scene",
            "face_crop",
            "x",
            "y",
            "width",
            "height",
            "confidence"
        ]
    )
    writer.writeheader()
    writer.writerows(manifest)

print("\nFace detection completed.")
print("Scenes processed:", len(files))
print("Faces detected:", face_id)
print("Face crops:", OUTPUT_DIR)
print("Manifest:", OUTPUT_CSV)
