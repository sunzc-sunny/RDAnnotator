import sys
import os
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

from get_annotation.rdannotator import AnnTool

if __name__ == '__main__':
    # Load paths from environment variables
    image_dir = os.getenv('NONCOLOR_IMAGE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/try_sample_image')
    caption_save_dir = os.getenv('NONCOLOR_CAPTION_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/color_caption_sample')
    color_check_save_dir = os.getenv('NONCOLOR_COLOR_CHECK_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/night_color_check_sample')
    color_annotator_save_dir = os.getenv('NONCOLOR_COLOR_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/night_color_annotation_sample')
    noncolor_annotator_save_dir = os.getenv('NONCOLOR_NONCOLOR_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/night_noncolor_annotation_sample')
    color_check_annotation_save_dir = os.getenv('NONCOLOR_COLOR_CHECK_ANNOTATION_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/night_color_check_annotation_sample')
    noncolor_check_annotation_save_dir = os.getenv('NONCOLOR_NONCOLOR_CHECK_ANNOTATION_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/night_noncolor_check_annotation_sample')
    color_regenerate_annotator_save_dir = os.getenv('NONCOLOR_COLOR_REGENERATE_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/night_color_regenerate_annotation_sample')
    noncolor_regenerate_annotator_save_dir = os.getenv('NONCOLOR_NONCOLOR_REGENERATE_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/night_noncolor_regenerate_annotation_sample')

    print(f"Using image_dir: {image_dir}")
    print(f"Using caption_save_dir: {caption_save_dir}")
    
    anntool = AnnTool(
        image_dir=image_dir, 
        caption_save_dir=caption_save_dir, 
        color_check_save_dir=color_check_save_dir, 
        color_annotator_save_dir=color_annotator_save_dir, 
        noncolor_annotator_save_dir=noncolor_annotator_save_dir, 
        color_check_annotation_save_dir=color_check_annotation_save_dir, 
        noncolor_check_annotation_save_dir=noncolor_check_annotation_save_dir,
        color_regenerate_annotator_save_dir=color_regenerate_annotator_save_dir, 
        noncolor_regenerate_annotator_save_dir=noncolor_regenerate_annotator_save_dir
    )
    anntool.noncolor_run(image_dir)


