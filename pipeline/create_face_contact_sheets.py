import csv
import os
import math
from PIL import Image, ImageDraw, ImageFont

INPUT_CSV = "faces/face_manifest.csv"
OUTPUT_DIR = "faces/contact_sheets"

COLUMNS = 10
FACES_PER_SHEET = 100
THUMB_SIZE = 100
LABEL_HEIGHT = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    faces = list(csv.DictReader(f))

font = ImageFont.load_default()

total_sheets = math.ceil(len(faces) / FACES_PER_SHEET)

for sheet_index in range(total_sheets):

    batch = faces[
        sheet_index * FACES_PER_SHEET:
        (sheet_index + 1) * FACES_PER_SHEET
    ]

    rows = math.ceil(len(batch) / COLUMNS)
    cell_height = THUMB_SIZE + LABEL_HEIGHT

    sheet = Image.new(
        "RGB",
        (COLUMNS * THUMB_SIZE, rows * cell_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, face in enumerate(batch):

        image = Image.open(face["face_crop"]).convert("RGB")
        image = image.resize((THUMB_SIZE, THUMB_SIZE))

        x = (index % COLUMNS) * THUMB_SIZE
        y = (index // COLUMNS) * cell_height

        sheet.paste(image, (x, y))

        label = (
            f"F{int(face['face_id']):04d} "
            f"S{face['scene'].replace('scene_', '').replace('.jpg', '')}"
        )

        draw.text(
            (x + 2, y + THUMB_SIZE + 3),
            label,
            fill="black",
            font=font
        )

    output = os.path.join(
        OUTPUT_DIR,
        f"face_sheet_{sheet_index + 1:03d}.jpg"
    )

    sheet.save(output, quality=85)

    print(f"Created {output}")

print("\nContact sheets completed.")
print("Total faces:", len(faces))
print("Total sheets:", total_sheets)
