import sys
sys.path.append('/home/sunzc/chatgpt/get_annotation/')
import base64
import requests
import random
import os
from concurrent.futures import ThreadPoolExecutor
import re
from PIL import Image

from io import BytesIO

from color_tools.regenerate_annotation_color import RegenerateAnnotatorColorV3

class BatchRegenerateAnnotatorNonColorV3(RegenerateAnnotatorColorV3):
    def __init__(self, prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, annotation_dir, n=1):
        super().__init__(prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, annotation_dir, n)

        self.system_message ={"role": "system", "content": f"""As an AI visual assistant, your role involves analyzing a single image. You are supplied with three sentences that caption the image, along with additional data about specific attributes of objects within the image. This can include information about categories and precise coordinates. Such coordinates, represented as floating-point numbers that range from 0 to 1, are shared as center points, denoted as (x, y), identifying the center x and y. When coordinate x tends to 0, the object nears the left side of the image, shifting towards the right as coordinate x approaches 1. When coordinate y tends to 0, the object nears the top of the image, shifting towards the bottom as coordinate y approaches 1. You are also provided with descriptions and the objects that initially failed to match, along with the reasons for the discrepancies.

Your task is to revise both the description and the corresponding objects to correct these mismatches based on the provided reasons. Ensure that the revised description accurately matches the corresponding objects depicted in the visual content.
"""}

    def create_json(self, image_name):
        system_message = self.system_message
        prompt_message = self.prompt_message
        query_message = self.get_query_message(image_name)
        message = [system_message]
        message.extend(prompt_message)
        message.append(query_message)
        return message


if __name__ == "__main__":
    from openai import OpenAI
    import json
    import io
    import time

    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
    client = OpenAI(api_key=api_key)

    image_dir = "/home/sunzc/VisDrone2019/visdrone_night_image_test"

    save_dir_new = "/home/sunzc/VisDroneAnnotation/test_noncolor_regenerate_annotation_others"
    save_dir_org = "/home/sunzc/VisDroneAnnotation/test_noncolor_regenerate_annotation_sample"
    os.makedirs(save_dir_new, exist_ok=True)
    os.makedirs(save_dir_org, exist_ok=True)

    ann_dir = "/home/sunzc/VisDroneAnnotation/test_noncolor_check_annotation_others"



    all_image_dir = "/data/sunzc/VisDrone2019/all_image"
    prompt_dir =  "/home/sunzc/chatgpt/prompts/regenerate_annotation_noncolor"

    caption_dir = "/home/sunzc/VisDroneAnnotation/test_color_caption_sample"

    color_info_dir = "/home/sunzc/VisDrone2019/test_anchor_normalize_noncolor"


    batch_regenerate_annotation_noncolor = BatchRegenerateAnnotatorNonColorV3(prompt_dir=prompt_dir, save_dir=save_dir_org, image_dir=image_dir, all_image_dir=all_image_dir, info_dir=color_info_dir, caption_dir=caption_dir, annotation_dir=ann_dir)
    


    image_names = os.listdir(image_dir)

    tasks = []
    batch_input_ids = []
    i = 0
    for image_name in image_names:
        if os.path.exists(os.path.join(save_dir_org, image_name.replace(".jpg", ".txt"))) or os.path.exists(os.path.join(save_dir_new, image_name.replace(".jpg", ".txt"))):
            continue

        json_file = {
            "custom_id": "",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {}
         }
        body_content = {
            "model": "gpt-4o",
            "temperature": 0.2,
            "top_p": 0.1,
            "presence_penalty": 2.0, 
            "n": 1,
            "messages": ""
         }
        json_file["custom_id"] = image_name
        cap_json = batch_regenerate_annotation_noncolor.create_json(image_name)
        
        if cap_json is None:
            continue
        i += 1
        body_content["messages"] = cap_json
        json_file["body"] = body_content
        tasks.append(json_file)


        if i == 60:
            json_str = ""
            for obj in tasks:
                json_str = json_str + (json.dumps(obj) + "\n")
            json_bytes = io.BytesIO(json_str.encode('utf-8'))
            batch_input_file = client.files.create(
                file=json_bytes,
                purpose="batch"
                )
            batch_input_file_id = batch_input_file.id

            batch_job = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h")

            batch_id = batch_job.id
            print(batch_id)

            batch_input_ids.append(batch_id)

            i = 0
            tasks = []
            time.sleep(10)

    json_str = ""
    for obj in tasks:
        json_str = json_str + (json.dumps(obj) + "\n")
    json_bytes = io.BytesIO(json_str.encode('utf-8'))

    batch_input_file = client.files.create(
    file=json_bytes,
    purpose="batch"
    )

    batch_input_file_id = batch_input_file.id



    batch_job = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h")

    batch_id = batch_job.id
    print(batch_id)
    batch_input_ids.append(batch_id)

    with open("batch_noncolor_regenerate_input_file_ids.txt", "w") as f:
        for batch_input_id in batch_input_ids:
            f.write(batch_input_id + "\n")


