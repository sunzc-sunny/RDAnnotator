from openai import OpenAI
import json
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY', 'your-api-key-here')

caption_save_dir = "/home/sunzc/VisDroneAnnotation/test_caption"
image_dir = "/data/sunzc/VisDrone2019/test-dev/images"


batch_file_dir = "/home/sunzc/chatgpt/batch_caption_input_file_ids_test.txt"

batch_list = []
with open(batch_file_dir, "r") as f:
    for line in f:
        batch_list.append(line.strip())


client = OpenAI(api_key=api_key)

batches = client.batches.list(limit=20)

batches = batches.json()

batches = batches.replace("null", "None").replace("false","False").replace("true","True")
batches = eval(batches)

i = 0
for item in batches["data"]:
    if item["id"] not in batch_list:
        print(f"skip {item['id']}")
        continue
    if item["status"] == "completed":
        print(item["output_file_id"], item["id"])
        file = client.files.content(item["output_file_id"]).content

        file_string = file.decode('utf-8')
        json_lines = file_string.split('\n')
        json_objects = [json.loads(line) for line in json_lines if line.strip()]
        for obj in json_objects:
            save_respones = ""

            custom_id = obj["custom_id"]
            if os.path.exists(os.path.join(image_dir, custom_id)) == False:
                continue

            save_annotation_path = os.path.join(caption_save_dir, custom_id.replace(".jpg", ".txt"))


            res_content = obj["response"]["body"]["choices"]
            for res in res_content:

                res = res['message']['content']
                if "\n" in res:
                    res = res.replace("\n", "")
                save_respones = save_respones + res + "\n\n"
            with open(save_annotation_path, "w") as f:
                f.write(save_respones)

            i += 1
print(f"total {i} images caption saved")
