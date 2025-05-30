# 用chatgpt 来判断要不要annotation是不是正确的


import base64
import requests
import random
import os
from concurrent.futures import ThreadPoolExecutor
import re                                                                                            



class CheckAnnotationNoncolor():
    def __init__(self, image_dir, info_dir, prompt_dir, save_dir, all_image_dir, caption_dir, annotation_dir, n=1):
        
        self.api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

        self.url = "https://api.openai.com/v1/chat/completions" 

        self.image_dir = image_dir
        self.info_dir = info_dir
        self.prompt_dir = prompt_dir
        self.save_dir = save_dir
        self.all_image_dir = all_image_dir
        self.caption_dir = caption_dir
        self.annotation_dir = annotation_dir   
        self.n = n
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"}

        self.payload = {
        "model": "gpt-4o",

        "messages": None,
        "temperature": 0.3,
        "top_p": 0.2,
        "n": self.n,
        }
        self.prompt_message = self.get_prompt(prompt_dir)

        self.system_message ={"role": "system", "content": f"""As an AI visual assistant, your role involves analyzing a single image. You are supplied with three sentences that describe the image, along with additional data about specific attributes of objects within the image. This can include information about categories and precise coordinates. Such coordinates, represented as floating-point numbers that range from 0 to 1, are shared as center points, denoted as (x, y), identifying the center x and y. When coordinate x tends to 0, the object nears the left side of the image, shifting towards the right as coordinate x approaches 1. When coordinate y tends to 0, the object nears the top of the image, shifting towards the bottom as coordinate y approaches 1. Besides, you are supplied with the description of the objects and their corresponding attributes.

Your task is to confirm whether the description exclusively relates to the described objects without including any others in the visual. Respond "yes" if it matches, or "no" with an explanation if it does not.
"""}


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
        annotation_file = os.path.join(self.annotation_dir, image_name.replace(".jpg", ".txt"))
        with open(bbox_file, "r") as f:
            bbox_info = f.read()
        with open(caption_file, "r") as f:
            caption_info = f.read()
        with open(annotation_file, "r") as f:
            annotation_info = f.read()


        info = "Captions:\n"+ caption_info + "\n" + "Objects:\n" + bbox_info + "\n\n" + "Descriptions:\n" + annotation_info

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




if __name__ == "__main__":
    image_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/10_1images"
    image_names = os.listdir(image_dir)
    
    prompt_dir = "/home/sunzc/chatgpt/prompts/check_annotation_example_noncolor"
    info_dir = "/home/sunzc/VisDrone2019/10_1images_annotation_noncolor"
    save_dir = "/home/sunzc/VisDrone2019/10_1check_annoaation_noncolor"
    all_image_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/images"
    caption_dir = "/home/sunzc/VisDrone2019/10_1images_captio_2"

    annotation_dir = "/home/sunzc/VisDrone2019/10_1_annoation_noncolor_v3"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_names = ["0000167_02184_d_0000135.jpg"]

    checkcolor = CheckAnnotationNoncolor(prompt_dir=prompt_dir, info_dir=info_dir, image_dir=image_dir, save_dir=save_dir, all_image_dir=all_image_dir, caption_dir=caption_dir, annotation_dir=annotation_dir, n=1)
    for image_name in image_names:
        responese = checkcolor.get_response(image_name)
        exit()

    max_workers = 10  
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(checkcolor.get_response, image_names)

