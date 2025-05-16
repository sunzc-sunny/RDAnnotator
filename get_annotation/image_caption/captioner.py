
import base64
import requests
import random
import os
from concurrent.futures import ThreadPoolExecutor
import re
from PIL import Image

from io import BytesIO

class Captioner():
    def __init__(self, image_dir, prompt_dir, save_dir, all_image_dir, n=3):
        # OpenAI API Key
        self.api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

        self.url = "https://api.openai.com/v1/chat/completions" 

        self.save_dir = save_dir
        self.image_dir = image_dir
        self.prompt_dir = prompt_dir
        self.all_image_dir = all_image_dir
        self.n = n
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"}

        self.payload = {
        "model": "gpt-4o",

        "messages": "",
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 2.0, 
        "n": self.n,
        }
        self.prompt_contents = [
        "Describe the image concisely.",
        "Provide a brief description of the given image.",
        "Offer a succinct explanation of the picture presented.",
        "Summarize the visual content of the image.",
        "Give a short and clear explanation of the subsequent image.",
        "Share a concise interpretation of the image provided.",
        "Present a compact description of the photo's key features.",
        "Relay a brief, clear account of the picture shown.",
        "Render a clear and concise summary of the photo.",
        "Write a terse but informative summary of the picture.",
        "Create a compact narrative representing the image presented."]

        self.prompt_message = self.get_prompt(prompt_dir)
        self.system_message = {"role": "system", "content": "You are an AI visual assistant that specializes in providing clear and accurate descriptions of images without any ambiguity or uncertainty. Your descriptions should focus solely on the content of the image itself and avoid mentioning any location-specific details such as regions or countries where the image might have been captured."}

    def encode_image(self, image_path):
        img = Image.open(image_path)
        width, height = img.size
        if width >= height:
            new_width = 512
            new_height = int(512 * height / width)
        else:
            new_height = 512
            new_width = int(512 * width / height)

        # 对图像进行缩放
        img_resized = img.resize((new_width, new_height))
        
        buffer = BytesIO()
        img_resized.save(buffer, format="JPEG") 
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return img_str


    def get_prompt(self, prompt_dir):
        prompt_files = os.listdir(prompt_dir)
        prompts = []

        for prompt_file in prompt_files:
            prompt_path = os.path.join(prompt_dir, prompt_file)
            prompt = {}
            with open(prompt_path, "r") as f:
                prompt_info = f.read()
                prompt["info"] = prompt_info
            prompt["text"] = random.choice(self.prompt_contents)
            name = prompt_file.replace(".txt", ".jpg")
            prompts_image = self.encode_image(os.path.join(self.all_image_dir, name))
            prompt["image"] = prompts_image
            prompts.append(prompt)

        prompt_message = []
        for prompt in prompts:
            prompt_info = prompt["info"]
            prompt_image = prompt["image"]
            prompt_text = prompt["text"]

            prompt_message.append({"role": "user", "content":
                [
                    {"type": "text", "text": str(prompt_text)},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{prompt_image}", "detail": "low"}}
                ]
            })
            prompt_message.append({"role": "assistant", "content": str(prompt_info)})


        return prompt_message


    def get_query_message(self, image_name):
        image_path = os.path.join(self.image_dir, image_name)
        base64_image = self.encode_image(image_path)
        prompt = random.choice(self.prompt_contents)

        query_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": str(prompt)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}}
            ]
        }

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
        print(res)
        save_annotation_path = os.path.join(self.save_dir, image_name.replace(".jpg", ".txt"))
        save_respones = ""
        if self.n == 1:
            res_content = res['choices'][0]['message']['content']
            save_respones = res_content
            with open(save_annotation_path, "w") as f:
                f.write(save_respones)
            print(f" save caption to {save_annotation_path}")

        else:

            for i in range(self.n):
                res_content = res['choices'][i]['message']['content']
                if "\n" in res_content:
                    res_content = res_content.replace("\n", "")
                save_respones = save_respones + res_content + "\n\n"
            with open(save_annotation_path, "w") as f:
                f.write(save_respones)
            print(f" save caption to {save_annotation_path}")

        return save_respones




if __name__ == "__main__":

    image_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/10_1images"
    save_dir = "/home/sunzc/VisDrone2019/10_1_annoaation_caption_v5"
    all_image_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/images"
    prompt_dir = "/home/sunzc/chatgpt/prompts/caption"

    captioner = Captioner(image_dir=image_dir, prompt_dir=prompt_dir, save_dir=save_dir, all_image_dir=all_image_dir)
    
    image_names = ["0000167_02184_d_0000135.jpg"]
    for image_name in image_names:
        captioner.get_response(image_name)
        exit()

