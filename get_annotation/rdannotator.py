import os
import logging
import time
from datetime import datetime
from typing import List, Optional

from .color_tools.color_annotation_v3 import ColorAnnotatorV3
from .color_tools.check_color import CheckColor
from .color_tools.check_annotation_chatgpt import CheckAnnotationColor
from .color_tools.regenerate_annotation_color import RegenerateAnnotatorColorV3

from .noncolor_tools.annotation_noncolor_v3 import AnnotatorNonColorV3
from .noncolor_tools.check_annotation_chatgpt_noncolor import CheckAnnotationNoncolor
from .noncolor_tools.regenerate_annotation_noncolor import RegenerateAnnotatorNonColorV3

from .image_caption.captioner import Captioner

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

    def __init__(self, image_dir: str, caption_save_dir: str, 
                color_check_save_dir: str, color_annotator_save_dir: str, 
                noncolor_annotator_save_dir: str, color_check_annotation_save_dir: str,
                noncolor_check_annotation_save_dir: str, color_regenerate_annotator_save_dir: str,
                noncolor_regenerate_annotator_save_dir: str, color_info_dir: Optional[str] = None):
        """Initialize the annotation tool with configurable paths.
        
        Args:
            image_dir: Directory containing input images
            caption_save_dir: Directory to save generated captions
            color_check_save_dir: Directory to save color check results
            color_annotator_save_dir: Directory to save color annotations
            noncolor_annotator_save_dir: Directory to save noncolor annotations
            color_check_annotation_save_dir: Directory to save color check annotation results
            noncolor_check_annotation_save_dir: Directory to save noncolor check annotation results
            color_regenerate_annotator_save_dir: Directory to save regenerated color annotations
            noncolor_regenerate_annotator_save_dir: Directory to save regenerated noncolor annotations
            color_info_dir: Directory containing color annotation info (optional, required for color processing)
        """
        # Base directories configuration
        self.data_root = os.getenv('VISDRONE_DATA_ROOT', './data')
        self.prompt_root = os.getenv('PROMPT_ROOT', './prompts')
        
        # Initialize paths
        self.image_dir = image_dir
        self.color_info_dir = color_info_dir
        self.all_image_dir = image_dir

        # Create all save directories
        self.caption_save_dir = caption_save_dir
        self.color_check_save_dir = color_check_save_dir
        self.color_annotator_save_dir = color_annotator_save_dir
        self.noncolor_annotator_save_dir = noncolor_annotator_save_dir
        self.color_check_annotation_save_dir = color_check_annotation_save_dir
        self.noncolor_check_annotation_save_dir = noncolor_check_annotation_save_dir
        self.color_regenerate_annotator_save_dir = color_regenerate_annotator_save_dir
        self.noncolor_regenerate_annotator_save_dir = noncolor_regenerate_annotator_save_dir
        
        # Create all directories
        for dir_path in [self.caption_save_dir, self.color_check_save_dir, 
                        self.color_annotator_save_dir, self.noncolor_annotator_save_dir,
                        self.color_check_annotation_save_dir, self.noncolor_check_annotation_save_dir,
                        self.color_regenerate_annotator_save_dir, self.noncolor_regenerate_annotator_save_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize color-related tools (only if color_info_dir is provided)
        self.checkcolor = None
        self.color_annotator = None
        self.color_check_annotation = None
        self.color_regenerate_annotator = None
        
        self.init_captioner()
        if color_info_dir is not None:
            self.init_checkcolor()
            self.init_color_annotator()
            self.init_check_annotation_color()
            self.init_regenerate_annotation_color()
        self.init_noncolor_annotator()
        self.init_check_annotation_noncolor()
        self.init_regenerate_annotation_noncolor()

        self.color_annotation_names = []
        self.noncolor_annotation_names = []
        self.others_annotation_names = []

    def init_captioner(self):
        self.caption_prompt_dir = os.path.join(self.prompt_root, 'caption')

        print(self.image_dir, self.all_image_dir)


        self.captioner = Captioner(image_dir=self.image_dir, prompt_dir=self.caption_prompt_dir, save_dir=self.caption_save_dir, all_image_dir=self.all_image_dir)
    
    def init_checkcolor(self):
        if self.color_info_dir is None:
            raise ValueError("color_info_dir is required for color processing")
        self.color_check_prompt_dir = os.path.join(self.prompt_root, 'check_color_example')

        self.checkcolor = CheckColor(prompt_dir=self.color_check_prompt_dir, info_dir=self.color_info_dir, image_dir=self.image_dir, save_dir=self.color_check_save_dir, all_image_dir=self.all_image_dir, n=1)
    
    def init_color_annotator(self):
        if self.color_info_dir is None:
            raise ValueError("color_info_dir is required for color processing")
        self.color_annotator_prompt_dir = os.path.join(self.prompt_root, 'annotation_example_color_v3')
        if not os.path.exists(self.color_annotator_prompt_dir):
            # Try project-relative path first, then fallback to env or default
            project_prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts', 'annotation_example_color_v3')
            if os.path.exists(project_prompt_dir):
                self.color_annotator_prompt_dir = project_prompt_dir
            else:
                self.color_annotator_prompt_dir = os.getenv('ANNOTATION_COLOR_V3_PROMPT_DIR', project_prompt_dir)
        self.color_annotator_info_dir = self.color_info_dir
        self.color_annotator_caption_dir = self.caption_save_dir

        self.color_annotator = ColorAnnotatorV3(prompt_dir=self.color_annotator_prompt_dir, info_dir=self.color_annotator_info_dir, caption_dir=self.color_annotator_caption_dir, image_dir=self.image_dir, save_dir=self.color_annotator_save_dir, all_image_dir=self.all_image_dir, n=1)

    def init_noncolor_annotator(self):
        self.noncolor_annotator_prompt_dir = os.path.join(self.prompt_root, 'annotation_example_noncolor_v3')

        # Use NONCOLOR_INFO_DIR from environment variable, with fallback options
        self.noncolor_annotator_info_dir = os.getenv('NONCOLOR_INFO_DIR')
        
        # Try multiple fallback paths
        fallback_paths = [
            # 1. Environment variable path
            self.noncolor_annotator_info_dir if self.noncolor_annotator_info_dir else None,
            # 2. Relative path from image_dir
            os.path.join(self.image_dir, '..', 'visdrone_anchor_normalize_noncolor'),
            # 3. Same directory level as image_dir
            os.path.join(os.path.dirname(self.image_dir), 'visdrone_anchor_normalize_noncolor'),
            # 4. all_annotations directory (common location)
            os.path.join(os.path.dirname(self.image_dir), 'all_annotations'),
            # 5. Alternative default path
            '/mnt/public/usr/sunzhichao/VisDrone2019/all_annotations',
        ]
        
        # Find the first existing directory
        found_dir = None
        for path in fallback_paths:
            if path and os.path.exists(path) and os.path.isdir(path):
                found_dir = path
                break
        
        if found_dir:
            self.noncolor_annotator_info_dir = found_dir
            if found_dir != os.getenv('NONCOLOR_INFO_DIR'):
                print(f"Info: Using auto-detected NONCOLOR_INFO_DIR: {self.noncolor_annotator_info_dir}")
        else:
            # If no directory found, use the env variable or default, but warn
            self.noncolor_annotator_info_dir = os.getenv('NONCOLOR_INFO_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/all_annotations')
            print(f"Warning: NONCOLOR_INFO_DIR '{self.noncolor_annotator_info_dir}' may not exist!")
            print(f"Please verify the path in .env file. Current image_dir: {self.image_dir}")
        
        print(f"Using NONCOLOR_INFO_DIR: {self.noncolor_annotator_info_dir}")
        
        self.noncolor_annotator_caption_dir = self.caption_save_dir
        
        self.noncolor_annotator = AnnotatorNonColorV3(prompt_dir=self.noncolor_annotator_prompt_dir, info_dir=self.noncolor_annotator_info_dir, image_dir=self.image_dir, save_dir=self.noncolor_annotator_save_dir, all_image_dir=self.all_image_dir, caption_dir=self.noncolor_annotator_caption_dir, n=1)
    
    def init_check_annotation_color(self):
        if self.color_info_dir is None:
            raise ValueError("color_info_dir is required for color processing")
        self.color_check_annotation_prompt_dir = os.path.join(self.prompt_root, 'check_annotation_example')
        if not os.path.exists(self.color_check_annotation_prompt_dir):
            # Try project-relative path first, then fallback to env or default
            project_prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts', 'check_annotation_example')
            if os.path.exists(project_prompt_dir):
                self.color_check_annotation_prompt_dir = project_prompt_dir
            else:
                self.color_check_annotation_prompt_dir = os.getenv('CHECK_ANNOTATION_PROMPT_DIR', project_prompt_dir)
        self.color_check_annotation_info_dir = self.color_info_dir
        self.color_check_annotation_caption_dir = self.caption_save_dir
        self.color_check_annotation_annotation_dir = self.color_annotator_save_dir

        self.color_check_annotation = CheckAnnotationColor(prompt_dir=self.color_check_annotation_prompt_dir, save_dir=self.color_check_annotation_save_dir, image_dir=self.image_dir, all_image_dir=self.all_image_dir, info_dir=self.color_check_annotation_info_dir, caption_dir=self.color_check_annotation_caption_dir, annotation_dir=self.color_check_annotation_annotation_dir, n=1)
    
    def init_check_annotation_noncolor(self):
        self.noncolor_check_annotation_prompt_dir = os.path.join(self.prompt_root, 'check_annotation_example_noncolor')
        if not os.path.exists(self.noncolor_check_annotation_prompt_dir):
            # Try project-relative path first, then fallback to env or default
            project_prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts', 'check_annotation_example_noncolor')
            if os.path.exists(project_prompt_dir):
                self.noncolor_check_annotation_prompt_dir = project_prompt_dir
            else:
                self.noncolor_check_annotation_prompt_dir = os.getenv('CHECK_ANNOTATION_NONCOLOR_PROMPT_DIR', project_prompt_dir)
        self.noncolor_check_annotation_info_dir = self.noncolor_annotator_info_dir
        self.noncolor_check_annotation_caption_dir = self.caption_save_dir
        self.noncolor_check_annotation_annotation_dir = self.noncolor_annotator_save_dir

        self.noncolor_check_annotation = CheckAnnotationNoncolor(prompt_dir=self.noncolor_check_annotation_prompt_dir, info_dir=self.noncolor_check_annotation_info_dir, image_dir=self.image_dir, save_dir=self.noncolor_check_annotation_save_dir, all_image_dir=self.all_image_dir, caption_dir=self.noncolor_check_annotation_caption_dir, annotation_dir=self.noncolor_check_annotation_annotation_dir , n=1)

    def init_regenerate_annotation_color(self):
        if self.color_info_dir is None:
            raise ValueError("color_info_dir is required for color processing")
        self.color_regenerate_annotator_prompt_dir = os.path.join(self.prompt_root, 'regenerate_annotation_color')
        if not os.path.exists(self.color_regenerate_annotator_prompt_dir):
            # Try project-relative path first, then fallback to env or default
            project_prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts', 'regenerate_annotation_color')
            if os.path.exists(project_prompt_dir):
                self.color_regenerate_annotator_prompt_dir = project_prompt_dir
            else:
                self.color_regenerate_annotator_prompt_dir = os.getenv('REGENERATE_ANNOTATION_COLOR_PROMPT_DIR', project_prompt_dir)
        self.color_regenerate_annotator_info_dir = self.color_info_dir
        self.color_regenerate_annotator_caption_dir = self.caption_save_dir
        self.color_regenerate_annotator_annotation_dir = self.color_check_annotation_save_dir
        
        self.color_regenerate_annotator = RegenerateAnnotatorColorV3(prompt_dir=self.color_regenerate_annotator_prompt_dir, info_dir=self.color_regenerate_annotator_info_dir, image_dir=self.image_dir, save_dir=self.color_regenerate_annotator_save_dir, all_image_dir=self.all_image_dir, caption_dir=self.color_regenerate_annotator_caption_dir, annotation_dir=self.color_regenerate_annotator_annotation_dir, n=1)
    
    def init_regenerate_annotation_noncolor(self):
        self.noncolor_regenerate_annotator_prompt_dir = os.path.join(self.prompt_root, 'regenerate_annotation_noncolor')
        if not os.path.exists(self.noncolor_regenerate_annotator_prompt_dir):
            # Try project-relative path first, then fallback to env or default
            project_prompt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts', 'regenerate_annotation_noncolor')
            if os.path.exists(project_prompt_dir):
                self.noncolor_regenerate_annotator_prompt_dir = project_prompt_dir
            else:
                self.noncolor_regenerate_annotator_prompt_dir = os.getenv('REGENERATE_ANNOTATION_NONCOLOR_PROMPT_DIR', project_prompt_dir)
        self.noncolor_regenerate_annotator_info_dir = self.noncolor_annotator_info_dir
        self.noncolor_regenerate_annotator_caption_dir = self.caption_save_dir
        self.noncolor_regenerate_annotator_annotation_dir = self.noncolor_check_annotation_save_dir
        
        self.noncolor_regenerate_annotator = RegenerateAnnotatorNonColorV3(prompt_dir=self.noncolor_regenerate_annotator_prompt_dir, info_dir=self.noncolor_regenerate_annotator_info_dir, image_dir=self.image_dir, save_dir=self.noncolor_regenerate_annotator_save_dir, all_image_dir=self.all_image_dir, caption_dir=self.noncolor_regenerate_annotator_caption_dir, annotation_dir=self.noncolor_regenerate_annotator_annotation_dir, n=1)
    
    def get_caption(self, image_name=None):
        """Get caption for a single image or all images.
        
        Args:
            image_name: If provided, process only this image. Otherwise process all images.
        """
        if image_name:
            image_names = [image_name]
        else:
            image_names = os.listdir(self.image_dir)
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for img_name in image_names:
            # Check if file has supported image extension
            if not any(img_name.endswith(ext) for ext in image_extensions):
                continue
            
            # Get file extension and replace accordingly
            file_ext = os.path.splitext(img_name)[1]
            caption_path = os.path.join(self.caption_save_dir, img_name.replace(file_ext, ".txt"))
            if os.path.exists(caption_path):
                continue

            try:
                self.captioner.get_response(img_name)
            except Exception as e:
                logging.error(f"get_caption error in {img_name}: {e}")
                raise

    def get_checkcolor(self, image_name=None):
        """Check color for a single image or all images.
        
        Args:
            image_name: If provided, process only this image. Otherwise process all images.
        """
        if self.checkcolor is None:
            raise ValueError("checkcolor is not initialized. color_info_dir is required for color processing.")
        
        if image_name:
            image_names = [image_name]
        else:
            image_names = os.listdir(self.image_dir)
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for img_name in image_names:
            # Check if file has supported image extension
            if not any(img_name.endswith(ext) for ext in image_extensions):
                continue
            
            # Get file extension and replace accordingly
            file_ext = os.path.splitext(img_name)[1]
            check_file = os.path.join(self.color_check_save_dir, img_name.replace(file_ext, ".txt"))
            if os.path.exists(check_file):
                continue
            try:
                self.checkcolor.get_response(img_name)
            except Exception as e:
                logging.error(f"get_checkcolor error in {img_name}: {e}")
                time.sleep(60)
                try:
                    self.checkcolor.get_response(img_name)
                except Exception as e2:
                    logging.error(f"get_checkcolor retry error in {img_name}: {e2}")
                    raise

    def split_color_noncolor(self):
        color_annotation_names = []
        noncolor_annotation_names = []
        others_annotation_names = []
        color_check_files = os.listdir(self.color_check_save_dir)
        for color_check_file in color_check_files:
            with open(os.path.join(self.color_check_save_dir, color_check_file), "r") as f:
                color_annotation = f.read()

            print('color_annotation: ', color_annotation)
            if "Yes" in color_annotation:
                color_annotation_names.append(color_check_file.replace(".txt", ".jpg"))
            elif "No" in color_annotation:
                noncolor_annotation_names.append(color_check_file.replace(".txt", ".jpg"))
            else:
                others_annotation_names.append(color_check_file.replace(".txt", ".jpg"))
                
        return color_annotation_names, noncolor_annotation_names, others_annotation_names
    
    def get_color_annotator(self, image_name=None):
        """Get color annotation for a single image or all color images.
        
        Args:
            image_name: If provided, process only this image. Otherwise process all color images.
        """
        if self.color_annotator is None:
            raise ValueError("color_annotator is not initialized. color_info_dir is required for color processing.")
        
        if image_name:
            image_names = [image_name]
        else:
            image_names = self.color_annotation_names
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for img_name in image_names:
            # Check if file has supported image extension
            if not any(img_name.endswith(ext) for ext in image_extensions):
                continue
            
            # Get file extension and replace accordingly
            file_ext = os.path.splitext(img_name)[1]
            image_path = os.path.join(self.color_annotator_save_dir, img_name.replace(file_ext, ".txt"))
            if os.path.exists(image_path):
                continue
            try:
                self.color_annotator.get_response(img_name)
            except Exception as e:
                logging.error(f"get_color_annotator error in {img_name}: {e}")
                time.sleep(60)
                try:
                    self.color_annotator.get_response(img_name)
                except Exception as e2:
                    logging.error(f"get_color_annotator retry error in {img_name}: {e2}")
                    raise

    def get_noncolor_annotator(self, image_name=None):
        """Get noncolor annotation for a single image or all noncolor images.
        
        Args:
            image_name: If provided, process only this image. Otherwise process all noncolor images.
        """
        if image_name:
            image_names = [image_name]
        else:
            image_names = self.noncolor_annotation_names
        
        for img_name in image_names:
            if not img_name.endswith('.jpg'):
                continue
            annotation_path = os.path.join(self.noncolor_annotator_save_dir, img_name.replace(".jpg", ".txt"))
            if os.path.exists(annotation_path):
                continue
            try:
                self.noncolor_annotator.get_response(img_name)
            except Exception as e:
                logging.error(f"get_noncolor_annotator error in {img_name}: {e}")
                raise

    def get_check_annotation_color(self, image_name=None):
        """Check color annotation for a single image or all color images.
        
        Args:
            image_name: If provided, process only this image. Otherwise process all color images.
        """
        if self.color_check_annotation is None:
            raise ValueError("color_check_annotation is not initialized. color_info_dir is required for color processing.")
        
        if image_name:
            image_names = [image_name]
        else:
            image_names = self.color_annotation_names
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for img_name in image_names:
            # Check if file has supported image extension
            if not any(img_name.endswith(ext) for ext in image_extensions):
                continue
            
            # Get file extension and replace accordingly
            file_ext = os.path.splitext(img_name)[1]
            check_path = os.path.join(self.color_check_annotation_save_dir, img_name.replace(file_ext, ".txt"))
            if os.path.exists(check_path):
                continue
            
            try:
                self.color_check_annotation.get_response(img_name)
            except Exception as e:
                logging.error(f"get_check_annotation_color error in {img_name}: {e}")
                time.sleep(60)
                try:
                    self.color_check_annotation.get_response(img_name)
                except Exception as e2:
                    logging.error(f"get_check_annotation_color retry error in {img_name}: {e2}")
                    raise

    def get_check_annotation_noncolor(self, image_name=None):
        """Check noncolor annotation for a single image or all noncolor images.
        
        Args:
            image_name: If provided, process only this image. Otherwise process all noncolor images.
        """
        if image_name:
            image_names = [image_name]
        else:
            image_names = self.noncolor_annotation_names
        
        for img_name in image_names:
            if not img_name.endswith('.jpg'):
                continue
            check_path = os.path.join(self.noncolor_check_annotation_save_dir, img_name.replace(".jpg", ".txt"))
            if os.path.exists(check_path):
                continue
            try:
                self.noncolor_check_annotation.get_response(img_name)
            except Exception as e:
                logging.error(f"get_check_annotation_noncolor error in {img_name}: {e}")
                raise

    def get_regenerate_annotation_color(self, image_name=None):
        """Regenerate color annotation for a single image or all color images.
        
        Args:
            image_name: If provided, process only this image. Otherwise process all color images.
        """
        if self.color_regenerate_annotator is None:
            raise ValueError("color_regenerate_annotator is not initialized. color_info_dir is required for color processing.")
        
        if image_name:
            image_names = [image_name]
        else:
            image_names = self.color_annotation_names
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        
        for img_name in image_names:
            # Check if file has supported image extension
            if not any(img_name.endswith(ext) for ext in image_extensions):
                continue
            
            # Get file extension and replace accordingly
            file_ext = os.path.splitext(img_name)[1]
            regenerate_path = os.path.join(self.color_regenerate_annotator_save_dir, img_name.replace(file_ext, ".txt"))
            if os.path.exists(regenerate_path):
                continue

            try:
                self.color_regenerate_annotator.get_response(img_name)
            except Exception as e:
                logging.error(f"get_regenerate_annotation_color error in {img_name}: {e}")
                time.sleep(60)
                try:
                    self.color_regenerate_annotator.get_response(img_name)
                except Exception as e2:
                    logging.error(f"get_regenerate_annotation_color retry error in {img_name}: {e2}")
                    raise

    def get_regenerate_annotation_noncolor(self, image_name=None):
        """Regenerate noncolor annotation for a single image or all noncolor images.
        
        Args:
            image_name: If provided, process only this image. Otherwise process all noncolor images.
        """
        if image_name:
            image_names = [image_name]
        else:
            image_names = self.noncolor_annotation_names
        
        for img_name in image_names:
            if not img_name.endswith('.jpg'):
                continue
            regenerate_path = os.path.join(self.noncolor_regenerate_annotator_save_dir, img_name.replace(".jpg", ".txt"))
            if os.path.exists(regenerate_path):
                continue
            try:
                self.noncolor_regenerate_annotator.get_response(img_name)
            except Exception as e:
                logging.error(f"get_regenerate_annotation_noncolor error in {img_name}: {e}")
                raise
    
    def process_single_noncolor_image(self, image_name):
        """Process a single noncolor image through the complete pipeline.
        
        Args:
            image_name: Name of the image file to process
        """
        print(f"\n{'='*60}")
        print(f"Processing image: {image_name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Generate caption
            print(f"[{image_name}] Step 1/4: Generating caption...")
            self.get_caption(image_name)
            print(f"[{image_name}] ✓ Caption generated")
            
            # Step 2: Generate noncolor annotation
            print(f"[{image_name}] Step 2/4: Generating noncolor annotation...")
            self.get_noncolor_annotator(image_name)
            print(f"[{image_name}] ✓ Noncolor annotation generated")
            
            # Step 3: Check noncolor annotation
            print(f"[{image_name}] Step 3/4: Checking noncolor annotation...")
            self.get_check_annotation_noncolor(image_name)
            print(f"[{image_name}] ✓ Noncolor annotation checked")
            
            # Step 4: Regenerate noncolor annotation if needed
            print(f"[{image_name}] Step 4/4: Regenerating noncolor annotation if needed...")
            self.get_regenerate_annotation_noncolor(image_name)
            print(f"[{image_name}] ✓ Noncolor annotation regeneration completed")
            
            print(f"[{image_name}] ✓ All steps completed successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {image_name}: {e}")
            print(f"[{image_name}] ✗ Error: {e}")
            return False
    
    def noncolor_run(self, non_color_path):
        """Process all noncolor images one by one through the complete pipeline."""
        print("="*60)
        print("Starting Noncolor Annotation Pipeline")
        print("="*60)
        
        # Get all noncolor image names
        self.noncolor_annotation_names = [f for f in os.listdir(non_color_path) if f.endswith('.jpg')]
        total_images = len(self.noncolor_annotation_names)
        
        print(f"Total images to process: {total_images}")
        print("="*60)
        
        success_count = 0
        fail_count = 0
        
        for idx, image_name in enumerate(self.noncolor_annotation_names, 1):
            print(f"\n[{idx}/{total_images}] Processing: {image_name}")
            if self.process_single_noncolor_image(image_name):
                success_count += 1
            else:
                fail_count += 1
        
        print("\n" + "="*60)
        print("Noncolor Annotation Pipeline Completed!")
        print(f"Total: {total_images}, Success: {success_count}, Failed: {fail_count}")
        print("="*60)

    def process_single_color_image(self, image_name):
        """Process a single color image through the complete pipeline.
        
        Args:
            image_name: Name of the image file to process
        """
        print(f"\n{'='*60}")
        print(f"Processing image: {image_name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Generate caption
            print(f"[{image_name}] Step 1/5: Generating caption...")
            self.get_caption(image_name)
            print(f"[{image_name}] ✓ Caption generated")
            
            # Step 2: Check color
            print(f"[{image_name}] Step 2/5: Checking color...")
            self.get_checkcolor(image_name)
            print(f"[{image_name}] ✓ Color checked")
            
            # Step 3: Determine if it's a color image
            # Get file extension and replace accordingly
            file_ext = os.path.splitext(image_name)[1]
            check_file = os.path.join(self.color_check_save_dir, image_name.replace(file_ext, ".txt"))
            is_color = False
            if os.path.exists(check_file):
                with open(check_file, "r") as f:
                    check_result = f.read()
                    is_color = "Yes" in check_result
            
            if not is_color:
                print(f"[{image_name}] ⚠ Not a color image, skipping color-specific steps")
                return True
            
            # Temporarily set color_annotation_names to include only this image
            # so that the annotation methods can process it
            original_color_names = self.color_annotation_names.copy() if hasattr(self, 'color_annotation_names') else []
            self.color_annotation_names = [image_name]
            
            try:
                # Step 3: Generate color annotation
                print(f"[{image_name}] Step 3/5: Generating color annotation...")
                self.get_color_annotator(image_name)
                print(f"[{image_name}] ✓ Color annotation generated")
                
                # Step 4: Check color annotation
                print(f"[{image_name}] Step 4/5: Checking color annotation...")
                self.get_check_annotation_color(image_name)
                print(f"[{image_name}] ✓ Color annotation checked")
                
                # Step 5: Regenerate color annotation if needed
                print(f"[{image_name}] Step 5/5: Regenerating color annotation if needed...")
                self.get_regenerate_annotation_color(image_name)
                print(f"[{image_name}] ✓ Color annotation regeneration completed")
            finally:
                # Restore original color_annotation_names
                self.color_annotation_names = original_color_names
            
            print(f"[{image_name}] ✓ All steps completed successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {image_name}: {e}")
            print(f"[{image_name}] ✗ Error: {e}")
            return False
    
    def color_run(self, color_path):
        """Process all color images one by one through the complete pipeline."""
        print("="*60)
        print("Starting Color Annotation Pipeline")
        print("="*60)
        
        # Get all image names (support multiple formats)
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        all_image_names = [f for f in os.listdir(color_path) if any(f.endswith(ext) for ext in image_extensions)]
        total_images = len(all_image_names)
        
        print(f"Total images to process: {total_images}")
        print("="*60)
        
        success_count = 0
        fail_count = 0
        
        for idx, image_name in enumerate(all_image_names, 1):
            print(f"\n[{idx}/{total_images}] Processing: {image_name}")
            if self.process_single_color_image(image_name):
                success_count += 1
            else:
                fail_count += 1
        
        print("\n" + "="*60)
        print("Color Annotation Pipeline Completed!")
        print(f"Total: {total_images}, Success: {success_count}, Failed: {fail_count}")
        print("="*60)

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