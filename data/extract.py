import json
import random
import string
import numpy as np
import os
from PIL import Image, ImageDraw, ImageOps

os.makedirs("temp_annot/",exist_ok=True)
os.makedirs("dataset/",exist_ok=True)

src_dir =  "temp_annot/"
dest_dir = "dataset/"

for file in os.listdir(src_dir):
    if not file.endswith('.json'):

        img_path = os.path.join(src_dir,file)
        json_path = os.path.join(src_dir,os.path.splitext(file)[0]+'.json')

        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)

            image_height = annotation['imageHeight']
            image_width = annotation['imageWidth']
            image = Image.open(img_path)


            for shape in annotation['shapes']:
                shape_type = shape['shape_type']
                group_id = shape['group_id']
                label = shape['label']
                coordinates = shape['points']
                cropped_image = Image.new('RGBA', (image_width, image_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(cropped_image)

                random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=15))


                int_coordinates = [(int(point[0]), int(point[1])) for point in coordinates]

                draw.polygon(int_coordinates, fill=(255, 255, 255, 255))


                if shape_type == 'polygon':
                    mask = ImageOps.invert(cropped_image.convert('L'))

                    masked_image = Image.new("1", image.size)
                    masked_image.paste(image, (0, 0), mask=cropped_image)
                    bbox = masked_image.getbbox()

                    if bbox:
                        cropped_image = masked_image.crop(bbox)
                        rectangular_image = Image.new("1", (cropped_image.width, cropped_image.height), (255, 255, 255))
                        rectangular_image.paste(cropped_image, (0, 0), cropped_image)
                        new_ImageName = os.path.join(dest_dir, f"{random_string}.jpg")
                        new_LabelName = os.path.join(dest_dir, f"{random_string}.txt")
                        rectangular_image.save(new_ImageName, format='JPEG', quality=100)
                        # open(new_LabelName, "w", encoding="utf-8").write(label)
                        open(new_LabelName, "w").write(label)


                if shape_type == 'rectangle':
                    x_coordinates = [point[0] for point in int_coordinates]
                    y_coordinates = [point[1] for point in int_coordinates]
                    left = min(x_coordinates)
                    top = min(y_coordinates)
                    right = max(x_coordinates)
                    bottom = max(y_coordinates)
                    cropped_image = image.crop((left, top, right, bottom))
                    new_ImageName = os.path.join(dest_dir, f"{random_string}.jpg")
                    new_LabelName = os.path.join(dest_dir, f"{random_string}.txt")
                    cropped_image.save(new_ImageName, format='JPEG', quality=100)
                    # open(new_LabelName, "w", encoding="utf-8").write(label)
                    open(new_LabelName, "w").write(label)



# 1 (1-bit pixels, black and white, stored with one pixel per byte)
# L (8-bit pixels, grayscale)
# RGB (3x8-bit pixels, true color)
# RGBA (4x8-bit pixels, true color with transparency mask)
