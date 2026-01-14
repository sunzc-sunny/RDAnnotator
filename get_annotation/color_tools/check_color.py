"""Color checking tool using GPT-4 vision API."""
import base64
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

class CheckColor():
    """Tool for verifying color attributes using GPT-4 vision API."""
    
    def __init__(self, prompt_dir: str, info_dir: str, image_dir: str, 
                save_dir: str, all_image_dir: str, n: int = 1):
        """Initialize the color checker.
        
        Args:
            prompt_dir: Directory containing prompt examples
            info_dir: Directory containing object info files
            image_dir: Directory containing input images
            save_dir: Directory to save verification results
            all_image_dir: Directory containing all images
            n: Number of responses to generate
        """
        # Read API configuration from environment variables
        self.api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
        self.url = os.getenv('OPENAI_API_URL', 'https://api.openai.com/v1/chat/completions')

        self.prompt_dir = prompt_dir
        self.info_dir = info_dir
        self.image_dir = image_dir
        self.save_dir = save_dir
        self.all_image_dir = all_image_dir
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
        self.system_message ={"role": "system", "content": f"""As an AI visual assistant, your role involves analyzing a single image. 

You are supplied with the specific attributes of objects within the image. This can include information about categories, colors, and precise coordinates. Such coordinates, represented as floating-point numbers that range from 0 to 1, are shared as center points, denoted as (x, y), identifying the center x and y.

Your task is to assess whether the given colors of specific objects match their appearance in the image. Respond with "Yes" when the colors are appropriate. In cases where the colors are deemed inappropriate, respond with a concise "No."
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
        with open(bbox_file, "r") as f:
            bbox_info = f.read()

        info =  bbox_info
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
    # Example configuration using environment variables
    data_root = os.getenv('VISDRONE_DATA_ROOT', './data')
    output_root = os.getenv('OUTPUT_ROOT', './output')
    prompt_root = os.getenv('PROMPT_ROOT', './prompts')
    
    image_dir = os.path.join(data_root, 'train/10_1images')
    prompt_dir = os.path.join(prompt_root, 'check_color_example')
    info_dir = os.path.join(data_root, 'annotations/color/v3')
    save_annotation_dir = os.path.join(output_root, 'color_check/v2')
    all_image_dir = os.path.join(data_root, 'train/images')

    if not os.path.exists(save_annotation_dir):
        os.makedirs(save_annotation_dir)

    image_names = os.listdir(image_dir)


    checkcolor = CheckColor(prompt_dir, info_dir, image_dir, save_annotation_dir, all_image_dir, n=1)

    max_workers = 10  
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(checkcolor.get_response, image_names)

