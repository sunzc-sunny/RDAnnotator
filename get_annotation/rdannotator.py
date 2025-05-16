import os
import logging
import time
from datetime import datetime
from typing import List, Optional

from color_tools.color_annotation_v3 import ColorAnnotatorV3
from color_tools.check_color import CheckColor
from color_tools.check_annotation_chatgpt import CheckAnnotationColor
from color_tools.regenerate_annotation_color import RegenerateAnnotatorColorV3

from noncolor_tools.annotation_noncolor_v3 import AnnotatorNonColorV3
from noncolor_tools.check_annotation_chatgpt_noncolor import CheckAnnotationNoncolor
from noncolor_tools.regenerate_annotation_noncolor import RegenerateAnnotatorNonColorV3

from image_caption.captioner import Captioner

# Configure logging to use project-relative paths
log_dir = os.getenv('LOG_DIR', './logs')
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f'error_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure logging
logging.basicConfig(
    filename=log_filename, 
    level=logging.ERROR, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)

class AnnTool():

    def __init__(self, image_dir: str, color_info_dir: str, caption_save_dir: str, 
                color_check_save_dir: str, color_annotator_save_dir: str, 
                noncolor_annotator_save_dir: str, color_check_annotation_save_dir: str,
                noncolor_check_annotation_save_dir: str, color_regenerate_annotator_save_dir: str,
                noncolor_regenerate_annotator_save_dir: str):
        """Initialize the annotation tool with configurable paths.
        
        Args:
            image_dir: Directory containing input images
            color_info_dir: Directory containing color annotation info
            caption_save_dir: Directory to save generated captions
            ... [other parameters]
        """
        # Base directories configuration
        self.data_root = os.getenv('VISDRONE_DATA_ROOT', './data')
        self.prompt_root = os.getenv('PROMPT_ROOT', './prompts')
        
        # Initialize paths
        self.image_dir = image_dir
        self.color_info_dir = color_info_dir
        self.all_image_dir = os.path.join(self.data_root, 'all_image')
        self.caption_save_dir = caption_save_dir
        self.color_check_save_dir = color_check_save_dir
        self.color_annotator_save_dir = color_annotator_save_dir
        self.noncolor_annotator_save_dir = noncolor_annotator_save_dir
        self.color_check_annotation_save_dir = color_check_annotation_save_dir
        self.noncolor_check_annotation_save_dir = noncolor_check_annotation_save_dir
        self.color_regenerate_annotator_save_dir = color_regenerate_annotator_save_dir
        self.noncolor_regenerate_annotator_save_dir = noncolor_regenerate_annotator_save_dir
        
        self.init_captioner()
        self.init_checkcolor()
        self.init_color_annotator()
        self.init_noncolor_annotator()
        self.init_check_annotation_color()
        self.init_check_annotation_noncolor()
        self.init_regenerate_annotation_color()
        self.init_regenerate_annotation_noncolor()

        self.color_annotation_names = []
        self.noncolor_annotation_names = []
        self.others_annotation_names = []

    def init_captioner(self):
        self.caption_prompt_dir = "/home/sunzc/chatgpt/prompts/caption"

        self.captioner = Captioner(image_dir=self.image_dir, prompt_dir=self.caption_prompt_dir, save_dir=self.caption_save_dir, all_image_dir=self.all_image_dir)
    
    def init_checkcolor(self):
        self.color_check_prompt_dir = "/home/sunzc/chatgpt/prompts/check_color_example"

        self.checkcolor = CheckColor(prompt_dir=self.color_check_prompt_dir, info_dir=self.color_info_dir, image_dir=self.image_dir, save_dir=self.color_check_save_dir, all_image_dir=self.all_image_dir, n=1)
    
    def init_color_annotator(self):
        self.color_annotator_prompt_dir = "/home/sunzc/chatgpt/prompts/annotation_example_color_v3"
        self.color_annotator_info_dir = self.color_info_dir
        self.color_annotator_caption_dir = self.caption_save_dir

        self.color_annotator = ColorAnnotatorV3(prompt_dir=self.color_annotator_prompt_dir, info_dir=self.color_annotator_info_dir, caption_dir=self.color_annotator_caption_dir, image_dir=self.image_dir, save_dir=self.color_annotator_save_dir, all_image_dir=self.all_image_dir, n=1)

    def init_noncolor_annotator(self):
        self.noncolor_annotator_prompt_dir = "/home/sunzc/chatgpt/prompts/annotation_example_noncolor_v3"
        self.noncolor_annotator_info_dir = "/home/sunzc/VisDrone2019/visdrone_anchor_normalize_noncolor"
        self.noncolor_annotator_caption_dir = self.caption_save_dir
        
        self.noncolor_annotator = AnnotatorNonColorV3(prompt_dir=self.noncolor_annotator_prompt_dir, info_dir=self.noncolor_annotator_info_dir, image_dir=self.image_dir, save_dir=self.noncolor_annotator_save_dir, all_image_dir=self.all_image_dir, caption_dir=self.noncolor_annotator_caption_dir, n=1)
    
    def init_check_annotation_color(self):
        self.color_check_annotation_prompt_dir = "/home/sunzc/chatgpt/prompts/check_annotation_example"
        self.color_check_annotation_info_dir = self.color_info_dir
        self.color_check_annotation_caption_dir = self.caption_save_dir
        self.color_check_annotation_annotation_dir = self.color_annotator_save_dir

        self.color_check_annotation = CheckAnnotationColor(prompt_dir=self.color_check_annotation_prompt_dir, save_dir=self.color_check_annotation_save_dir, image_dir=self.image_dir, all_image_dir=self.all_image_dir, info_dir=self.color_check_annotation_info_dir, caption_dir=self.color_check_annotation_caption_dir, annotation_dir=self.color_check_annotation_annotation_dir, n=1)
    
    def init_check_annotation_noncolor(self):
        self.noncolor_check_annotation_prompt_dir = "/home/sunzc/chatgpt/prompts/check_annotation_example_noncolor"
        self.noncolor_check_annotation_info_dir = self.noncolor_annotator_info_dir
        self.noncolor_check_annotation_caption_dir = self.caption_save_dir
        self.noncolor_check_annotation_annotation_dir = self.noncolor_annotator_save_dir

        self.noncolor_check_annotation = CheckAnnotationNoncolor(prompt_dir=self.noncolor_check_annotation_prompt_dir, info_dir=self.noncolor_check_annotation_info_dir, image_dir=self.image_dir, save_dir=self.noncolor_check_annotation_save_dir, all_image_dir=self.all_image_dir, caption_dir=self.noncolor_check_annotation_caption_dir, annotation_dir=self.noncolor_check_annotation_annotation_dir , n=1)

    def init_regenerate_annotation_color(self):
        self.color_regenerate_annotator_prompt_dir = "/home/sunzc/chatgpt/prompts/regenerate_annotation_color"
        self.color_regenerate_annotator_info_dir = self.color_info_dir
        self.color_regenerate_annotator_caption_dir = self.caption_save_dir
        self.color_regenerate_annotator_annotation_dir = self.color_check_annotation_save_dir
        
        self.color_regenerate_annotator = RegenerateAnnotatorColorV3(prompt_dir=self.color_regenerate_annotator_prompt_dir, info_dir=self.color_regenerate_annotator_info_dir, image_dir=self.image_dir, save_dir=self.color_regenerate_annotator_save_dir, all_image_dir=self.all_image_dir, caption_dir=self.color_regenerate_annotator_caption_dir, annotation_dir=self.color_regenerate_annotator_annotation_dir, n=1)
    
    def init_regenerate_annotation_noncolor(self):
        self.noncolor_regenerate_annotator_prompt_dir = "/home/sunzc/chatgpt/prompts/regenerate_annotation_noncolor"
        self.noncolor_regenerate_annotator_info_dir = self.noncolor_annotator_info_dir
        self.noncolor_regenerate_annotator_caption_dir = self.caption_save_dir
        self.noncolor_regenerate_annotator_annotation_dir = self.noncolor_check_annotation_save_dir
        
        self.noncolor_regenerate_annotator = RegenerateAnnotatorNonColorV3(prompt_dir=self.noncolor_regenerate_annotator_prompt_dir, info_dir=self.noncolor_regenerate_annotator_info_dir, image_dir=self.image_dir, save_dir=self.noncolor_regenerate_annotator_save_dir, all_image_dir=self.all_image_dir, caption_dir=self.noncolor_regenerate_annotator_caption_dir, annotation_dir=self.noncolor_regenerate_annotator_annotation_dir, n=1)
    
    def get_caption(self):
        image_names = os.listdir(self.image_dir)
        for image_name in image_names:
            
            image_dir = os.path.join(self.caption_save_dir, image_name.replace(".jpg", ".txt"))
            if os.path.exists(image_dir):
                continue

            try:
                self.captioner.get_response(image_name)
            except Exception as e:
                logging.error(f"get_caption error in {image_name}: {e}")

    def get_checkcolor(self):
        image_names = os.listdir(self.image_dir)
        for image_name in image_names:
            self.checkcolor.get_response(image_name)

    def split_color_noncolor(self):
        color_annotation_names = []
        noncolor_annotation_names = []
        others_annotation_names = []
        color_check_files = os.listdir(self.color_check_save_dir)
        for color_check_file in color_check_files:
            with open(os.path.join(self.color_check_save_dir, color_check_file), "r") as f:
                color_annotation = f.read()
            if "Yes" in color_annotation:
                color_annotation_names.append(color_check_file.replace(".txt", ".jpg"))
            elif "No" in color_annotation:
                noncolor_annotation_names.append(color_check_file.replace(".txt", ".jpg"))
            else:
                others_annotation_names.append(color_check_file.replace(".txt", ".jpg"))
                
        return color_annotation_names, noncolor_annotation_names, others_annotation_names
    
    def get_color_annotator(self):
        image_names = self.color_annotation_names
        for image_name in image_names:
            image_dir = os.path.join(self.color_annotator_save_dir, image_name.replace(".jpg", ".txt"))
            if os.path.exists(image_dir):
                continue
            try:
                self.color_annotator.get_response(image_name)
            except Exception as e:
                logging.error(f"get_color_annotator error in {image_name}: {e}")
                time.sleep(60)
                self.color_annotator.get_response(image_name)

    def get_noncolor_annotator(self):
        image_names = self.noncolor_annotation_names
        for image_name in image_names:
            self.noncolor_annotator.get_response(image_name)

    def get_check_annotation_color(self):
        image_names = self.color_annotation_names
        for image_name in image_names:
            
            image_dir = os.path.join(self.color_check_annotation_save_dir, image_name.replace(".jpg", ".txt"))
            if os.path.exists(image_dir):
                continue
            
            try:
                self.color_check_annotation.get_response(image_name)
            except Exception as e:
                logging.error(f"get_check_annotation_color error in {image_name}: {e}")
                time.sleep(60)
                self.color_check_annotation.get_response(image_name)

    def get_check_annotation_noncolor(self):
        image_names = self.noncolor_annotation_names
        for image_name in image_names:
            self.noncolor_check_annotation.get_response(image_name)

    def get_regenerate_annotation_color(self):
        image_names = self.color_annotation_names
        for image_name in image_names:

            image_dir = os.path.join(self.color_regenerate_annotator_save_dir, image_name.replace(".jpg", ".txt"))
            if os.path.exists(image_dir):
                continue

            try:
                self.color_regenerate_annotator.get_response(image_name)
            except Exception as e:
                logging.error(f"get_regenerate_annotation_color error in {image_name}: {e}")
                time.sleep(60)
                self.color_regenerate_annotator.get_response(image_name)

    def get_regenerate_annotation_noncolor(self):
        image_names = self.noncolor_annotation_names
        for image_name in image_names:
            self.noncolor_regenerate_annotator.get_response(image_name)
    
    def noncolor_run(self, non_color_path):
        # step 1: caption generation
        self.get_caption()

        # step 2: split color and noncolor
        self.noncolor_annotation_names = os.listdir(non_color_path)
        # step 3: noncolor annotation
        self.get_noncolor_annotator()

        # step 4: check noncolor annotation
        self.get_check_annotation_noncolor()

        # step 5: regenerate noncolor annotation
        self.get_regenerate_annotation_noncolor()

    def color_run(self, color_path):
        # step 1: caption generation
        self.get_caption()
        # step 2: check color
        self.get_checkcolor()
        # step 3: split color and noncolor
        self.color_annotation_names, self.noncolor_annotation_names, self.others_annotation_names = self.split_color_noncolor()        
        self.color_annotation_names = os.listdir(color_path)
        # step 4: color annotation
        self.get_color_annotator()
        # step 5: check color annotation
        self.get_check_annotation_color()
        # step 6: regenerate color annotation
        self.get_regenerate_annotation_color()

    def run(self):
        # step 1: caption generation
        self.get_caption()
        # step 2: check color
        self.color_annotation_names, self.noncolor_annotation_names, self.others_annotation_names = self.split_color_noncolor()

        # step 4: color annotation
        self.get_color_annotator()
        # step 5: check color annotation
        self.get_check_annotation_color()
        # step 6: regenerate color annotation
        self.get_regenerate_annotation_color()

        # step 7: noncolor annotation
        self.get_noncolor_annotator()
        # step 8: check noncolor annotation
        self.get_check_annotation_noncolor()
        # step 9: regenerate noncolor annotation
        self.get_regenerate_annotation_noncolor()



if __name__ == "__main__":



    # Example configuration using environment variables
    data_root = os.getenv('VISDRONE_DATA_ROOT', './data')
    output_root = os.getenv('OUTPUT_ROOT', './output')
    
    image_dir = os.path.join(data_root, 'sample_images/night')
    caption_save_dir = os.path.join(output_root, 'captions/night_sample')
    color_check_save_dir = os.path.join(output_root, 'color_check/night_sample')
    color_annotator_save_dir = os.path.join(output_root, 'color_annotation/night_sample')
    noncolor_annotator_save_dir = os.path.join(output_root, 'noncolor_annotation/night_sample')
    color_check_annotation_save_dir = os.path.join(output_root, 'color_check_annotation/night_sample')
    noncolor_check_annotation_save_dir = os.path.join(output_root, 'noncolor_check_annotation/night_sample')
    color_regenerate_annotator_save_dir = os.path.join(output_root, 'color_regenerate_annotation/night_sample')
    noncolor_regenerate_annotator_save_dir = os.path.join(output_root, 'noncolor_regenerate_annotation/night_sample')

    color_info_dir = os.path.join(data_root, 'annotations/color/wideresnet101')

    anntool = AnnTool(image_dir, color_info_dir=color_info_dir, caption_save_dir=caption_save_dir, color_check_save_dir=color_check_save_dir, color_annotator_save_dir=color_annotator_save_dir, 
    noncolor_annotator_save_dir=noncolor_annotator_save_dir, color_check_annotation_save_dir=color_check_annotation_save_dir, noncolor_check_annotation_save_dir=noncolor_check_annotation_save_dir,
    color_regenerate_annotator_save_dir=color_regenerate_annotator_save_dir, noncolor_regenerate_annotator_save_dir=noncolor_regenerate_annotator_save_dir)
    anntool.noncolor_run()