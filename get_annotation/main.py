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

def run_color_classification():
    """Step 1: Run color classification to generate color and noncolor info files."""
    print("="*60)
    print("Step 1: Running Color Classification")
    print("="*60)
    
    # Import and run get_color.py
    import subprocess
    color_classification_script = os.path.join(project_root, 'color_classification', 'get_color.py')
    
    if not os.path.exists(color_classification_script):
        raise FileNotFoundError(f"Color classification script not found: {color_classification_script}")
    
    print(f"Running: {color_classification_script}")
    result = subprocess.run([sys.executable, color_classification_script], 
                          cwd=os.path.dirname(color_classification_script),
                          capture_output=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Color classification failed with return code {result.returncode}")
    
    print("✓ Color classification completed")
    print("="*60)

def process_images_with_checkcolor():
    """Step 2-3: Process images using checkcolor and route to color/noncolor pipeline."""
    print("="*60)
    print("Step 2-3: Processing Images with Color Check")
    print("="*60)
    
    # Load paths from environment variables
    image_dir = os.getenv('IMAGE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/all_image')
    color_info_dir = os.getenv('COLOR_INFO_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/color_info')
    noncolor_info_dir = os.getenv('NONCOLOR_INFO_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/noncolor_info')
    
    caption_save_dir = os.getenv('CAPTION_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/caption')
    color_check_save_dir = os.getenv('COLOR_CHECK_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/color_check')
    color_annotator_save_dir = os.getenv('COLOR_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/color_annotation')
    noncolor_annotator_save_dir = os.getenv('NONCOLOR_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/noncolor_annotation')
    color_check_annotation_save_dir = os.getenv('COLOR_CHECK_ANNOTATION_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/color_check_annotation')
    noncolor_check_annotation_save_dir = os.getenv('NONCOLOR_CHECK_ANNOTATION_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/noncolor_check_annotation')
    color_regenerate_annotator_save_dir = os.getenv('COLOR_REGENERATE_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/color_regenerate_annotation')
    noncolor_regenerate_annotator_save_dir = os.getenv('NONCOLOR_REGENERATE_ANNOTATOR_SAVE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/VisDroneAnnotation/noncolor_regenerate_annotation')
    
    print(f"Using image_dir: {image_dir}")
    print(f"Using color_info_dir: {color_info_dir}")
    print(f"Using noncolor_info_dir: {noncolor_info_dir}")
    
    # Validate required directories
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory does not exist: {image_dir}")
    
    if not os.path.exists(color_info_dir):
        raise ValueError(f"Color info directory does not exist: {color_info_dir}. Please run color classification first.")
    
    if not os.path.exists(noncolor_info_dir):
        raise ValueError(f"Noncolor info directory does not exist: {noncolor_info_dir}. Please run color classification first.")
    
    # Initialize AnnTool
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
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_image_names = [f for f in os.listdir(image_dir) if any(f.endswith(ext) for ext in image_extensions)]
    total_images = len(all_image_names)
    
    print(f"\nTotal images to process: {total_images}")
    print("="*60)
    
    # Step 2: Generate captions for all images
    print("\nStep 2.1: Generating captions for all images...")
    for idx, image_name in enumerate(all_image_names, 1):
        print(f"[{idx}/{total_images}] Generating caption for {image_name}...")
        try:
            anntool.get_caption(image_name)
        except Exception as e:
            print(f"Error generating caption for {image_name}: {e}")
            continue
    
    print("✓ Caption generation completed")
    
    # Step 2.2: Check color for all images
    print("\nStep 2.2: Checking color for all images...")
    for idx, image_name in enumerate(all_image_names, 1):
        print(f"[{idx}/{total_images}] Checking color for {image_name}...")
        try:
            anntool.get_checkcolor(image_name)
        except Exception as e:
            print(f"Error checking color for {image_name}: {e}")
            continue
    
    print("✓ Color check completed")
    
    # Step 3: Split images into color and noncolor based on check results
    print("\nStep 3: Splitting images into color and noncolor groups...")
    color_image_names, noncolor_image_names, others_image_names = anntool.split_color_noncolor()
    
    print(f"Color images: {len(color_image_names)}")
    print(f"Noncolor images: {len(noncolor_image_names)}")
    print(f"Others/Uncertain: {len(others_image_names)}")
    
    # Step 4: Process color images
    if color_image_names:
        print("\n" + "="*60)
        print("Step 4: Processing Color Images")
        print("="*60)
        anntool.color_annotation_names = color_image_names
        
        for idx, image_name in enumerate(color_image_names, 1):
            print(f"\n[{idx}/{len(color_image_names)}] Processing color image: {image_name}")
            try:
                # Generate color annotation
                anntool.get_color_annotator(image_name)
                # Check color annotation
                anntool.get_check_annotation_color(image_name)
                # Regenerate if needed
                anntool.get_regenerate_annotation_color(image_name)
            except Exception as e:
                print(f"Error processing color image {image_name}: {e}")
                continue
    
    # Step 5: Process noncolor images (including those with color check problems)
    if noncolor_image_names or others_image_names:
        print("\n" + "="*60)
        print("Step 5: Processing Noncolor Images")
        print("="*60)
        
        # Combine noncolor and others (uncertain) images
        all_noncolor_images = noncolor_image_names + others_image_names
        anntool.noncolor_annotation_names = all_noncolor_images
        
        # Update noncolor_info_dir for noncolor annotator and related tools
        # This ensures it uses the noncolor info files we generated
        anntool.noncolor_annotator_info_dir = noncolor_info_dir
        anntool.noncolor_check_annotation_info_dir = noncolor_info_dir
        anntool.noncolor_regenerate_annotator_info_dir = noncolor_info_dir
        
        # Reinitialize noncolor tools with updated info_dir
        from get_annotation.noncolor_tools.annotation_noncolor_v3 import AnnotatorNonColorV3
        from get_annotation.noncolor_tools.check_annotation_chatgpt_noncolor import CheckAnnotationNoncolor
        from get_annotation.noncolor_tools.regenerate_annotation_noncolor import RegenerateAnnotatorNonColorV3
        
        anntool.noncolor_annotator = AnnotatorNonColorV3(
            prompt_dir=anntool.noncolor_annotator_prompt_dir,
            info_dir=noncolor_info_dir,
            image_dir=anntool.image_dir,
            save_dir=anntool.noncolor_annotator_save_dir,
            all_image_dir=anntool.all_image_dir,
            caption_dir=anntool.caption_save_dir,
            n=1
        )
        
        anntool.noncolor_check_annotation = CheckAnnotationNoncolor(
            prompt_dir=anntool.noncolor_check_annotation_prompt_dir,
            info_dir=noncolor_info_dir,
            image_dir=anntool.image_dir,
            save_dir=anntool.noncolor_check_annotation_save_dir,
            all_image_dir=anntool.all_image_dir,
            caption_dir=anntool.caption_save_dir,
            annotation_dir=anntool.noncolor_annotator_save_dir,
            n=1
        )
        
        anntool.noncolor_regenerate_annotator = RegenerateAnnotatorNonColorV3(
            prompt_dir=anntool.noncolor_regenerate_annotator_prompt_dir,
            info_dir=noncolor_info_dir,
            image_dir=anntool.image_dir,
            save_dir=anntool.noncolor_regenerate_annotator_save_dir,
            all_image_dir=anntool.all_image_dir,
            caption_dir=anntool.caption_save_dir,
            annotation_dir=anntool.noncolor_check_annotation_save_dir,
            n=1
        )
        
        print(f"Using noncolor_info_dir: {noncolor_info_dir}")
        
        for idx, image_name in enumerate(all_noncolor_images, 1):
            print(f"\n[{idx}/{len(all_noncolor_images)}] Processing noncolor image: {image_name}")
            try:
                # Generate noncolor annotation
                anntool.get_noncolor_annotator(image_name)
                # Check noncolor annotation
                anntool.get_check_annotation_noncolor(image_name)
                # Regenerate if needed
                anntool.get_regenerate_annotation_noncolor(image_name)
            except Exception as e:
                print(f"Error processing noncolor image {image_name}: {e}")
                continue
    
    print("\n" + "="*60)
    print("All Processing Completed!")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Color images: {len(color_image_names)}")
    print(f"Noncolor images: {len(noncolor_image_names + others_image_names)}")
    print("="*60)

if __name__ == '__main__':
    try:
        # Step 1: Run color classification
        run_color_classification()
        
        # Step 2-5: Process images with checkcolor and route accordingly
        process_images_with_checkcolor()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

