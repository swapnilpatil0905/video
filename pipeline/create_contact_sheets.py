import os
import math
from PIL import Image, ImageDraw, ImageFont

INPUT_DIR = "scenes/frames"
OUTPUT_DIR = "scenes/contact_sheets"

THUMB_WIDTH = 120
COLUMNS = 10
FRAMES_PER_SHEET = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

font = ImageFont.load_default()

for sheet_index in range(math.ceil(len(files) / FRAMES_PER_SHEET)):
    batch = files[
        sheet_index * FRAMES_PER_SHEET:
        (sheet_index + 1) * FRAMES_PER_SHEET
    ]

    thumbnails = []

    for filename in batch:
        path = os.path.join(INPUT_DIR, filename)

        with Image.open(path) as img:
            img = img.convert("RGB")
            thumb_height = int(img.height * THUMB_WIDTH / img.width)
            img = img.resize((THUMB_WIDTH, thumb_height))

        thumbnails.append((filename, img))

    thumb_height = thumbnails[0][1].height
    label_height = 22
    cell_height = thumb_height + label_height

    rows = math.ceil(len(thumbnails) / COLUMNS)

    sheet = Image.new(
        "RGB",
        (COLUMNS * THUMB_WIDTH, rows * cell_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, (filename, img) in enumerate(thumbnails):
        x = (index % COLUMNS) * THUMB_WIDTH
        y = (index // COLUMNS) * cell_height

        sheet.paste(img, (x, y))
        draw.text(
            (x + 3, y + thumb_height + 3),
            filename.replace(".jpg", ""),
            fill="black",
            font=font
        )

    output = os.path.join(
        OUTPUT_DIR,
        f"sheet_{sheet_index + 1:03d}.jpg"
    )

    sheet.save(output, quality=85)

    print(f"Created {output}")

print("\nContact sheets completed.")
print("Total frames:", len(files))
print("Total sheets:", math.ceil(len(files) / FRAMES_PER_SHEET))
