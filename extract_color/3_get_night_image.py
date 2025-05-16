# Extracts night images from VisDrone2019 dataset based on pixel brightness
from PIL import Image
import os
import shutil


# Input/output paths
image_root = "data/visdrone_color_image_val"  # Directory containing color images
save_night_image = "output/visdrone_night_image_val"  # Output directory for night images



if not os.path.exists(save_night_image):
    os.makedirs(save_night_image)

image_names = os.listdir(image_root)


for image_name in image_names:
    image_path = os.path.join(image_root, image_name)
    image = Image.open(image_path).convert('RGB')
    average_pixel = tuple(map(int, image.resize((1, 1), Image.LANCZOS).getpixel((0, 0))))

    # Calculate brightness by averaging pixel values (excluding the brightest channel)
    average_pixel = list(average_pixel)
    max_pixel = max(average_pixel)
    average_pixel = sum(average_pixel) - max_pixel
    if average_pixel <= 120:
        print(f"{image_name} is a night image")
        shutil.move(image_path, os.path.join(save_night_image, image_name))

