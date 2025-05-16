"""Tool for regenerating color annotations using GPT-4 vision API."""
import base64
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

from .color_annotation_v3 import ColorAnnotatorV3

class RegenerateAnnotatorColorV3(ColorAnnotatorV3):
    """Tool for regenerating color annotations based on verification feedback."""
    
    def __init__(self, prompt_dir: str, info_dir: str, image_dir: str, 
                save_dir: str, all_image_dir: str, caption_dir: str,
                annotation_dir: str, n: int = 1):
        """Initialize the annotation regenerator.
        
        Args:
            prompt_dir: Directory containing prompt examples
            info_dir: Directory containing object info files
            image_dir: Directory containing input images
            save_dir: Directory to save regenerated annotations
            all_image_dir: Directory containing all images
            caption_dir: Directory containing image captions
            annotation_dir: Directory containing annotations to regenerate
            n: Number of responses to generate
        """
        super().__init__(prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, n)
        self.annotation_dir = annotation_dir
        self.system_message ={"role": "system", "content": f"""As an AI visual assistant, your role involves analyzing a single image. You are supplied with three sentences that caption the image, along with additional data about specific attributes of objects within the image. This can include information about categories, colors, and precise coordinates. Such coordinates, represented as floating-point numbers that range from 0 to 1, are shared as center points, denoted as (x, y), identifying the center x and y. When coordinate x tends to 0, the object nears the left side of the image, shifting towards the right as coordinate x approaches 1. When coordinate y tends to 0, the object nears the top of the image, shifting towards the bottom as coordinate y approaches 1. You are also provided with descriptions and the objects that initially failed to match, along with the reasons for the discrepancies.

Your task is to revise both the description and the corresponding objects to correct these mismatches based on the provided reasons. Ensure that the revised description accurately matches the corresponding objects depicted in the visual content.
"""}

    
    def get_query_message(self, image_name):
        bbox_file = os.path.join(self.info_dir, image_name.replace(".jpg", ".txt"))
        caption_file = os.path.join(self.caption_dir, image_name.replace(".jpg", ".txt"))
        annotation_file = os.path.join(self.annotation_dir, image_name.replace(".jpg", ".txt")) 

        with open(bbox_file, "r") as f:
            bbox_info = f.read()

        with open(caption_file, "r") as f:
            caption_info = f.read()

        with open(annotation_file, "r") as f:
            annotation_info = f.read()
        failed_contents = []
        annotation_contents = annotation_info.split("\n\n")
        for annotation_content in annotation_contents:
            if "No" in annotation_content:
                failed_contents.append(annotation_content)

        if failed_contents == []:
            return None

        # for failed_content in failed_contents:
        new_failed_contents = []
        for i in range(len(failed_contents)):
            failed_content = failed_contents[i]
            failed_content  = failed_content.replace("No, ", "Reason:\n")
            failed_content  = failed_content.replace("No", "Reason:\n")

            try:
                words = failed_content.split("Reason:\n")[1].split()
            except:
                print(failed_content)
                words = ""
            content = failed_content.split("Reason:\n")[0]
            if words:
                words[0] = words[0].capitalize()
                new_reason = ' '.join(words)
                new_failed_content = "Description:\n" + content + "\n\nReason:\n" + new_reason
                new_failed_contents.append(new_failed_content)


            

        info = "Captions:\n"+ caption_info + "\n" + "Objects:\n" + bbox_info
        for new_failed_content in new_failed_contents:
            info += "\n\n" + new_failed_content + "\n"

        image_path = os.path.join(self.all_image_dir, image_name)
        base64_image = self.encode_image(image_path)
        query_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": str(info),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            }
            ]}
        return query_message

 
    def get_response(self, image_name):
        system_message = self.system_message
        prompt_message = self.prompt_message
        query_message = self.get_query_message(image_name)
        if query_message is None:
            return None
            
        message = [system_message]
        message.extend(prompt_message)
        message.append(query_message)
        self.payload["messages"] = message

        response = requests.post(self.url, headers=self.headers, json=self.payload)
        res = response.json()
        save_annotation_path = os.path.join(self.save_dir, image_name.replace(".jpg", ".txt"))
        save_respones = None
        if self.n == 1:
            print(res)
            res_content = res['choices'][0]['message']['content']
            save_respones = res_content
            print(save_respones)
            with open(save_annotation_path, "w") as f:
                f.write(save_respones)
            print(f" save caption to {save_annotation_path}")

        else:

            for i in range(self.n):
                res_content = res['choices'][i]['message']['content']
                save_respones = save_respones + res_content + "\n\n"
            with open(save_annotation_path, "w") as f:
                f.write(save_respones)
            print(f" save caption to {save_annotation_path}")

        return save_respones



if __name__ == "__main__":
    data_root = os.getenv('VISDRONE_DATA_ROOT', './data')
    output_root = os.getenv('OUTPUT_ROOT', './output')
    prompt_root = os.getenv('PROMPT_ROOT', './prompts')
    
    image_dir = os.path.join(data_root, 'train/10_1images')
    prompt_dir = os.path.join(prompt_root, 'regenerate_annotation_color')
    info_dir = os.path.join(data_root, 'annotations/color')
    save_dir = os.path.join(output_root, 'regenerated_annotations/v3')
    all_image_dir = os.path.join(data_root, 'train/images')
    caption_dir = os.path.join(output_root, 'captions/10_1')
    annotation_dir = os.path.join(output_root, 'annotation_check')



    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_names = os.listdir(image_dir)



    checkcolor = RegenerateAnnotatorColorV3(prompt_dir=prompt_dir, info_dir=info_dir, image_dir=image_dir, save_dir=save_dir, all_image_dir=all_image_dir, caption_dir=caption_dir, annotation_dir=annotation_dir, n=1)
    print(checkcolor.prompt_message)
    for image_name in image_names:
        responese = checkcolor.get_response(image_name)
        exit()

    max_workers = 10  # 设置线程池中的最大工作线程数量
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(checkcolor.get_response, image_names)