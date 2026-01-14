from openai import OpenAI
import json
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

image_dir = "/home/sunzc/VisDrone2019/visdrone_color_image_test"
save_dir = "/home/sunzc/VisDroneAnnotation/test_color_annotation_others33"
os.makedirs(save_dir, exist_ok=True)


file_dir = "/home/sunzc/chatgpt/get_annotation/color_tools/batch_1yunXgZZiozgkOPsAVRCL4UP_output.jsonl"
i = 0
file = open(file_dir, "r")
json_lines = file.readlines()
json_objects = [json.loads(line) for line in json_lines if line.strip()]

for obj in json_objects:
    save_respones = ""

    custom_id = obj["custom_id"]
    if os.path.exists(os.path.join(image_dir, custom_id)) == False:
        continue

    save_annotation_path = os.path.join(save_dir, custom_id.replace(".jpg", ".txt"))


    res_content = obj["response"]["body"]["choices"]
    for res in res_content:

        res = res['message']['content']

        save_respones = save_respones + res + "\n\n"
    with open(save_annotation_path, "w") as f:
        f.write(save_respones)
    print(f" save color annotation to {save_annotation_path}")
    i += 1
print(f"total {i} images caption saved")
