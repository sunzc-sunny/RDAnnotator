
import base64
import requests
import random
import os
from concurrent.futures import ThreadPoolExecutor
import re
from PIL import Image
from .color_annotation_v3 import ColorAnnotatorV3

from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

class BatchColorAnnotatorV3(ColorAnnotatorV3):
    def __init__(self, prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, n=1):
        super().__init__(prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, n)


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


    image_dir = "/home/sunzc/VisDrone2019/visdrone_color_image_val"
    save_dir_org = "/home/sunzc/VisDroneAnnotation/val_color_annotation"

    save_dir_new = "/home/sunzc/VisDroneAnnotation/val_color_annotation"

    if not os.path.exists(save_dir_org):
        os.makedirs(save_dir_org)
    if not os.path.exists(save_dir_new):
        os.makedirs(save_dir_new)

    all_image_dir = "/data/sunzc/VisDrone2019/all_image"
    prompt_dir = "/home/sunzc/chatgpt/prompts/annotation_example_color_v3"

    caption_dir = "/home/sunzc/VisDroneAnnotation/val_caption"


    color_info_dir = "/home/sunzc/VisDrone2019/val_image_color"



    batchcolorannotator = BatchColorAnnotatorV3(prompt_dir=prompt_dir, info_dir=color_info_dir, caption_dir=caption_dir, image_dir=image_dir, save_dir=save_dir_new, all_image_dir=all_image_dir, n=1)


    image_names = os.listdir(image_dir)

    tasks = []
    batch_input_ids = []
    i = 0
    for image_name in image_names:
        if os.path.exists(os.path.join(save_dir_org, image_name.replace(".jpg", ".txt"))) or os.path.exists(os.path.join(save_dir_new, image_name.replace(".jpg", ".txt"))):
            continue

        i += 1
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
        cap_json = batchcolorannotator.create_json(image_name)
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

    # print("\n\n")
    # print(tasks)
    # exit()
    json_str = ""
    for obj in tasks:
        json_str = json_str + (json.dumps(obj) + "\n")
    # json_str = (json.dumps(obj) + "\n") for obj in tasks
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

    # 把所有的batch_input_file_id保存到文件中
    with open("batch_caption_input_file_ids_train.txt", "w") as f:
        for batch_input_id in batch_input_ids:
            f.write(batch_input_id + "\n")


