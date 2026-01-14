import sys
import os
import torch
import torch.nn as nn
import torchvision.models as models
from vocr_dataset import VCoR
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

class CustomWideResNet101(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomWideResNet101, self).__init__()
        # Load pre-trained WideResNet101-2 model
        self.wide_resnet = models.wide_resnet101_2(pretrained=False)
        
        # Modify the final fully connected layer to output 6 classes
        in_features = self.wide_resnet.fc.in_features
        self.wide_resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.wide_resnet(x)

if __name__ == '__main__':
    # Load paths from environment variables
    image_root = os.getenv('IMAGE_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/all_image')
    ann_root = os.getenv('ANNOTATION_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/all_annotations')
    color_info_dir = os.getenv('COLOR_INFO_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/color_info')
    noncolor_info_dir = os.getenv('NONCOLOR_INFO_DIR', '/mnt/public/usr/sunzhichao/VisDrone2019/noncolor_info')
    os.makedirs(color_info_dir, exist_ok=True)
    os.makedirs(noncolor_info_dir, exist_ok=True)
    # Model file path (relative to script directory)
    model_file = os.path.join(script_dir, 'wide_resnet_color_model_99_89.80.pth')
    
    print(f"Using image_root: {image_root}")
    print(f"Using ann_root: {ann_root}")
    print(f"Using color_info_dir: {color_info_dir}")
    print(f"Using noncolor_info_dir: {noncolor_info_dir}")
    print(f"Using model_file: {model_file}")
    
    # Validate required directories
    if not os.path.exists(image_root):
        raise ValueError(f"Image directory does not exist: {image_root}")
    
    if not os.path.exists(ann_root):
        raise ValueError(f"Annotation directory does not exist: {ann_root}")
    
    if not os.path.exists(model_file):
        raise ValueError(f"Model file does not exist: {model_file}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(color_info_dir):
        os.makedirs(color_info_dir, exist_ok=True)
        print(f"Created color_info_dir: {color_info_dir}")
    
    # Initialize model
    model = CustomWideResNet101()
    
    # Load checkpoint
    checkpoint = torch.load(model_file, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    color_mapping = {
        0: 'black', 1: 'blue', 2: 'green', 3: 'red', 4: 'white', 5: 'yellow'
    }
    
    transform = transforms.Compose([
        transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    class_name_list = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Get all image files
    image_names = [f for f in os.listdir(image_root) if any(f.endswith(ext) for ext in image_extensions)]
    total_images = len(image_names)
    
    print(f"Found {total_images} images to process")
    print("="*60)
    
    processed_count = 0
    error_count = 0
    
    for idx, image_name in enumerate(image_names, 1):
        try:
            # Get file extension
            file_ext = os.path.splitext(image_name)[1]
            image_path = os.path.join(image_root, image_name)
            ann_name = os.path.splitext(image_name)[0] + '.txt'
            ann_path = os.path.join(ann_root, ann_name)
            
            # Skip if annotation file doesn't exist
            if not os.path.exists(ann_path):
                print(f"[{idx}/{total_images}] Skipping {image_name}: annotation file not found")
                continue
            
            # Skip if output files already exist
            save_color_ann_path = os.path.join(color_info_dir, ann_name)
            save_noncolor_ann_path = os.path.join(noncolor_info_dir, ann_name)
            if os.path.exists(save_color_ann_path) and os.path.exists(save_noncolor_ann_path):
                print(f"[{idx}/{total_images}] Skipping {image_name}: output files already exist")
                continue
            
            image = Image.open(image_path).convert('RGB')
            image_size = image.size
            
            print(f"[{idx}/{total_images}] Processing {image_name}...")
            
            # Paths for color and noncolor output files
            save_color_ann_path = os.path.join(color_info_dir, ann_name)
            save_noncolor_ann_path = os.path.join(noncolor_info_dir, ann_name)
            
            with open(save_color_ann_path, "w") as f_color_out, open(save_noncolor_ann_path, "w") as f_noncolor_out:
                with open(ann_path, "r") as f:
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue
                        
                        line_parts = line.split(',')
                        if len(line_parts) < 6:
                            continue
                        
                        try:
                            x1, y1, w, h = map(int, line_parts[:4])
                            class_num = int(line_parts[5])
                            
                            # Skip small objects
                            if w * h <= 128:
                                continue
                            
                            # Skip ignored regions and others
                            if class_num == 0 or class_num == 11:
                                continue
                            
                            # Calculate bounding box with padding
                            bbox = [max(0, x1-4), max(0, y1-4), 
                                   min(image_size[0], x1 + w - 4), 
                                   min(image_size[1], y1 + h - 4)]
                            
                            # Skip invalid bboxes
                            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                                continue
                            
                            # Calculate normalized center point
                            center_point = [(x1 + 0.5 * w), (y1 + 0.5 * h)]
                            normal_center_point = [
                                round(center_point[0] / image_size[0], 3), 
                                round(center_point[1] / image_size[1], 3)
                            ]
                            
                            class_name = class_name_list[class_num]
                            
                            # Crop object region
                            object_region = image.crop(bbox)
                            
                            # Transform and predict
                            object_region_tensor = transform(object_region)
                            object_region_tensor = object_region_tensor.unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                outputs = model(object_region_tensor)
                                _, predicted = torch.max(outputs.data, 1)
                            
                            # Write color result: "class_name, color: [x, y]"
                            color_str = f"{class_name}, {color_mapping[predicted.item()]}: [{normal_center_point[0]}, {normal_center_point[1]}]"
                            f_color_out.write(color_str + "\n")
                            
                            # Write noncolor result: "class_name: [x, y]"
                            noncolor_str = f"{class_name}: [{normal_center_point[0]}, {normal_center_point[1]}]"
                            f_noncolor_out.write(noncolor_str + "\n")
                            
                        except (ValueError, IndexError) as e:
                            # Skip invalid annotation lines
                            continue
            
            processed_count += 1
            if idx % 100 == 0:
                print(f"Progress: {idx}/{total_images} images processed")
                
        except Exception as e:
            error_count += 1
            print(f"Error processing {image_name}: {e}")
            continue
    
    print("="*60)
    print(f"Processing completed!")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Color info output directory: {color_info_dir}")
    print(f"Noncolor info output directory: {noncolor_info_dir}")



