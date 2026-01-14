import base64
import requests
import random
import os
from concurrent.futures import ThreadPoolExecutor
import re                                                                                            
from dotenv import load_dotenv
load_dotenv()

from ..color_tools.regenerate_annotation_color import RegenerateAnnotatorColorV3

class RegenerateAnnotatorNonColorV3(RegenerateAnnotatorColorV3):
    def __init__(self, prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, annotation_dir, n=1):
        super().__init__(prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, annotation_dir, n)



        self.system_message ={"role": "system", "content": f"""As an AI visual assistant, your role involves analyzing a single image. You are supplied with three sentences that caption the image, along with additional data about specific attributes of objects within the image. This can include information about categories and precise coordinates. Such coordinates, represented as floating-point numbers that range from 0 to 1, are shared as center points, denoted as (x, y), identifying the center x and y. When coordinate x tends to 0, the object nears the left side of the image, shifting towards the right as coordinate x approaches 1. When coordinate y tends to 0, the object nears the top of the image, shifting towards the bottom as coordinate y approaches 1. You are also provided with descriptions and the objects that initially failed to match, along with the reasons for the discrepancies.

Your task is to revise both the description and the corresponding objects to correct these mismatches based on the provided reasons. Ensure that the revised description accurately matches the corresponding objects depicted in the visual content.
"""}

if __name__ == "__main__":
    image_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/10_1images"
    image_names = os.listdir(image_dir)
    
    prompt_dir = "/home/sunzc/chatgpt/prompts/regenerate_annotation_noncolor"
    info_dir = "/home/sunzc/VisDrone2019/10_1images_annotation_noncolor"
    save_dir = "/home/sunzc/VisDrone2019/10_1_reannoation_noncolor_v3"
    all_image_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/images"
    caption_dir = "/home/sunzc/VisDrone2019/10_1images_captio_2"
    annotation_dir = "/home/sunzc/VisDrone2019/10_1regenerate_annotation_noncolor"


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_names = ["0000167_02184_d_0000135.jpg"]



    checkcolor = RegenerateAnnotatorNonColorV3(prompt_dir=prompt_dir, info_dir=info_dir, image_dir=image_dir, save_dir=save_dir, all_image_dir=all_image_dir, caption_dir=caption_dir, annotation_dir=annotation_dir, n=1)
    for image_name in image_names:
        responese = checkcolor.get_response(image_name)
        exit()

    max_workers = 10  
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(checkcolor.get_response, image_names)