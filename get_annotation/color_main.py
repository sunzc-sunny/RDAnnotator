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
    image_dir = os.getenv('COLOR_IMAGE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/test_color_image_sample')
    color_info_dir = os.getenv('COLOR_INFO_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/test_color_image_sample_ann')
    
    caption_save_dir = os.getenv('COLOR_CAPTION_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/test_color_caption_sample')
    color_check_save_dir = os.getenv('COLOR_COLOR_CHECK_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/test_color_check_sample')
    color_annotator_save_dir = os.getenv('COLOR_COLOR_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/test_color_annotation_sample')
    noncolor_annotator_save_dir = os.getenv('COLOR_NONCOLOR_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/test_noncolor_annotation_sample')
    color_check_annotation_save_dir = os.getenv('COLOR_COLOR_CHECK_ANNOTATION_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/test_color_check_annotation_sample')
    noncolor_check_annotation_save_dir = os.getenv('COLOR_NONCOLOR_CHECK_ANNOTATION_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/test_noncolor_check_annotation_sample')
    color_regenerate_annotator_save_dir = os.getenv('COLOR_COLOR_REGENERATE_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/test_color_regenerate_annotation_sample')
    noncolor_regenerate_annotator_save_dir = os.getenv('COLOR_NONCOLOR_REGENERATE_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/test_noncolor_regenerate_annotation_sample')

    print(f"Using image_dir: {image_dir}")
    print(f"Using color_info_dir: {color_info_dir}")
    print(f"Using caption_save_dir: {caption_save_dir}")
    
    # Validate required directories
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    if not os.path.exists(color_info_dir):
        print(f"Warning: Color info directory does not exist: {color_info_dir}")
        print("Please ensure color information files are generated using color_classification/get_color.py")
        print("Or update COLOR_INFO_DIR in your .env file to point to the correct directory.")
        # Optionally create the directory if it doesn't exist
        # os.makedirs(color_info_dir, exist_ok=True)
    
    anntool = AnnTool(
        image_dir=image_dir, 
        color_info_dir=color_info_dir,
        caption_save_dir=caption_save_dir, 
        color_check_save_dir=color_check_save_dir, 
        color_annotator_save_dir=color_annotator_save_dir, 
        noncolor_annotator_save_dir=noncolor_annotator_save_dir, 
        color_check_annotation_save_dir=color_check_annotation_save_dir, 
        noncolor_check_annotation_save_dir=noncolor_check_annotation_save_dir,
        color_regenerate_annotator_save_dir=color_regenerate_annotator_save_dir, 
        noncolor_regenerate_annotator_save_dir=noncolor_regenerate_annotator_save_dir
    )
    anntool.color_run(image_dir)


