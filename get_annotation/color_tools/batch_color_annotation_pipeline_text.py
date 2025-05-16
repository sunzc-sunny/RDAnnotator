
import base64
import requests
import random
import os
from concurrent.futures import ThreadPoolExecutor
import re
from PIL import Image
from color_annotation_v3 import ColorAnnotatorV3

from io import BytesIO
import json
from collections import defaultdict
import pycocotools.coco as coco



def get_questions(questions_path):
    organized_data = defaultdict(lambda: {'quetsions': []})
    data = coco.COCO(questions_path)
    old_caption = ""
    for item in data.getAnnIds():
        ann = data.anns[item]
        image = data.loadImgs(ann["image_id"])[0]
        file_name = image["file_name"]
        caption = image["caption"]
        if caption == old_caption:
            continue
        else:
            old_caption = caption
            organized_data[file_name]['quetsions'].append(caption)

    return organized_data


class TestColorAnnotatorV3():
    def __init__(self, prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, questions_path, n=1):


        self.api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

        self.url = "https://api.openai.com/v1/chat/completions" 
        


        self.image_dir = image_dir
        self.info_dir = info_dir
        self.prompt_dir = prompt_dir
        self.save_dir = save_dir
        self.caption_dir = caption_dir
        self.all_image_dir = all_image_dir
        self.questions_path = questions_path
        self.n = n
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"}

        self.payload = {
        "model": "gpt-4o",


        "messages": None,
        "temperature": 0.2,
        "top_p": 0.1,
        "presence_penalty": 2.0, 
        "n": self.n,
        }
        self.prompt_message = self.get_prompt(prompt_dir)
        self.system_message ={"role": "system", "content": f"""As an AI visual assistant, your role involves analyzing a single image. You are supplied with three sentences that caption the image, along with additional data about specific attributes of objects within the image. This can include information about categories, colors, and precise coordinates. Such coordinates, represented as floating-point numbers that range from 0 to 1, are shared as center points, denoted as (x, y), identifying the center x and y. When coordinate x tends to 0, the object nears the left side of the image, shifting towards the right as coordinate x approaches 1. When coordinate y tends to 0, the object nears the top of the image, shifting towards the bottom as coordinate y approaches 1.

Your task is to answer questions using the provided object coordinates, based on your analysis of the image, captions, and object data.
"""}
        self.question_data = get_questions(questions_path)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def get_prompt(self, prompt_dir):
        prompt_files = os.listdir(prompt_dir)
        prompts = []
        for prompt_file in prompt_files:
            if prompt_file.endswith("_info.txt"):
                prompt = {}
                with open(os.path.join(prompt_dir, prompt_file), "r") as f:
                    prompt_info = f.read()
                    prompt["info"] = prompt_info
                with open(os.path.join(prompt_dir, prompt_file.replace("_info.txt", "_ann.txt")), "r") as f:
                    prompt_text = f.read()
                    prompt["ann"] = prompt_text
                prompt_name = prompt_file.replace("_info.txt", ".jpg")
                prompt["name"] = prompt_name
                prompts_image = self.encode_image(os.path.join(self.all_image_dir, prompt_name))
                prompt["image"] = prompts_image
                prompts.append(prompt)

        prompt_message = []
        for prompt in prompts:
            prompt_info = prompt["info"]
            prompt_ann = prompt["ann"]
            prompt_image = prompt["image"]
            prompt_image1_name = prompt["name"]

            prompt_message.append({"role": "user", "content":
            [
                {
                    "type": "text",
                    "text": str(prompt_info)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{prompt_image}",
                        "detail": "high"
                    }
                }
            ]
            })
            prompt_message.append({"role": "assistant", "content": str(prompt_ann)})
        return prompt_message


    def get_query_message(self, image_name):
        bbox_file = os.path.join(self.info_dir, image_name.replace(".jpg", ".txt"))
        caption_file = os.path.join(self.caption_dir, image_name.replace(".jpg", ".txt"))
        question_info = self.question_data[image_name]
        question_info = question_info['quetsions']
        with open(bbox_file, "r") as f:
            bbox_info = f.read()

        with open(caption_file, "r") as f:
            caption_info = f.read()

        info = "Captions:\n"+ caption_info + "\n" + "Objects:\n" + bbox_info + "\n\n" + "Questions:\n" + "\n\n".join(question_info)

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
        message = [system_message]
        message.extend(prompt_message)
        message.append(query_message)
        self.payload["messages"] = message

        response = requests.post(self.url, headers=self.headers, json=self.payload)
        res = response.json()
        save_annotation_path = os.path.join(self.save_dir, image_name.replace(".jpg", ".txt"))
        save_respones = None
        if self.n == 1:
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


class BatchColorAnnotatorV3(TestColorAnnotatorV3):
    def __init__(self, prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, questions_path, n=1):
        super().__init__(prompt_dir, info_dir, image_dir, save_dir, all_image_dir, caption_dir, questions_path, n)


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


    image_dir = "/data/sunzc/VisDrone2019/test-dev/images"
    save_dir_org = "/home/sunzc/VisDroneAnnotation/pipeline_test_color_annotation_org_label"

    save_dir_new = "/home/sunzc/VisDroneAnnotation/pipeline_test_color_annotation_org_label"

    if not os.path.exists(save_dir_org):
        os.makedirs(save_dir_org)
    if not os.path.exists(save_dir_new):
        os.makedirs(save_dir_new)

    all_image_dir = "/data/sunzc/VisDrone2019/all_image"
    prompt_dir = "/home/sunzc/chatgpt/prompts/annotation_example_color_v3_pipeline_test"

    caption_dir = "/home/sunzc/VisDroneAnnotation/test_caption"


    color_info_dir = "/home/sunzc/VisDrone2019/test_image_color"


    questions_path = "/data/sunzc/RefDrone/finetune_RefDrone_test6.json"


    batchcolorannotator = BatchColorAnnotatorV3(prompt_dir=prompt_dir, info_dir=color_info_dir, caption_dir=caption_dir, 
                                                image_dir=image_dir, save_dir=save_dir_new, all_image_dir=all_image_dir, 
                                                questions_path=questions_path, n=1)

    question_data = batchcolorannotator.question_data



    tasks = []
    batch_input_ids = []
    i = 0
    for image_name, _ in question_data.items():
        if image_name.replace(".jpg", ".txt") not in os.listdir(color_info_dir):
            continue
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

    with open("batch_pipeline_test.txt", "w") as f:
        for batch_input_id in batch_input_ids:
            f.write(batch_input_id + "\n")


