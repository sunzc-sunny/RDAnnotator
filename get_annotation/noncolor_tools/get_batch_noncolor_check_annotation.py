from openai import OpenAI
import json
import os

api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
image_dir = "/home/sunzc/VisDrone2019/visdrone_night_image_test"


save_dir = "/home/sunzc/VisDroneAnnotation/test_noncolor_check_annotation_others"

os.makedirs(save_dir, exist_ok=True)

batch_file_dir = "/home/sunzc/chatgpt/get_annotation/noncolor_tools/batch_noncolor_check_input_file_ids.txt"

client = OpenAI(api_key=api_key)

batches = client.batches.list()

batches = batches.json()

batches = batches.replace("null", "None").replace("false","False").replace("true","True")
batches = eval(batches)

batch_list = []
with open(batch_file_dir, "r") as f:
    for line in f:
        batch_list.append(line.strip())


i = 0
for item in batches["data"]:
    if item["id"] not in batch_list:
        print(f"skip {item['id']}")
        continue
    if item["status"] == "completed":
        print(item["output_file_id"])
        file = client.files.content(item["output_file_id"]).content
        file_string = file.decode('utf-8')
        json_lines = file_string.split('\n')
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
