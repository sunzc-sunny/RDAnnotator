# 1. Merge corrected annotations with original annotations to create new annotations
# 2. Restore bounding boxes for the new annotations and save as final results

# 3. Visualize annotation results with the following specifications:
#    - Objects referenced in text descriptions are marked in red
#    - Other objects of the same category are marked in green  
#    - Display center coordinates at red point locations
#    - Add a subtitle area below the image for text descriptions

import os
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt





image_annotation_orginal = '/data/sunzc/RefDrone/test_7_12'


image_annotation_check = "/home/sunzc/VisDroneAnnotation/test_color_check_annotation_others"

image_reannotation_annotation = "/home/sunzc/VisDroneAnnotation/test_color_regenerate_annotation_others"

image_path = "/data/sunzc/VisDrone2019/all_image"


ann_path = '/home/sunzc/VisDroneAnnotation/test_image_color_ann_others'
ann_path = '/home/sunzc/VisDrone2019/test_color_image_sample_ann'

save_path = '/home/sunzc/VisDroneAnnotation/test_color_rerecheck_p2'

if os.path.exists(save_path) is False:
    os.makedirs(save_path)


ann_image_names = os.listdir(image_annotation_orginal)


def put_text_with_wrap(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, font_thickness=2, max_line_width=None, line_spacing=10, color=(255, 255, 255)):
    lines = []
    font_height = cv2.getTextSize('A', font, font_scale, font_thickness)[0][1]
    max_width = max_line_width if max_line_width else img.shape[1] - position[0] - 20
    words = text.split()
    line = ''
    for word in words:
        test_line = f'{line} {word}'.strip()
        width = cv2.getTextSize(test_line, font, font_scale, font_thickness)[0][0]
        if width < max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)
    y = position[1]
    for line in lines:
        cv2.putText(img, line, (position[0], y), font, font_scale, color, font_thickness)
        y += font_height + line_spacing
    return img, y

def vis_image(ann_image_name):

    check_states = []
    image_check_dir = os.path.join(image_annotation_check, ann_image_name)
    if image_check_dir.split('.')[-1] != "txt":
        return
    with open(image_check_dir, "r") as f:
        check_infos = f.read()
    if check_infos.endswith('\n'):
        check_infos = check_infos[:-2]
    print(check_infos)
    check_infos = check_infos.split('\n\n')
    for check_info in check_infos:
        check_state = check_info.split('\n')[-1]
        print(check_state)
        if check_state == "Yes":
            check_states.append(0)
        else:
            check_states.append(1)

    ann_image_dir = os.path.join(image_annotation_orginal, ann_image_name)
    with open(ann_image_dir, "r") as f: 
        ann_infos = f.read()
    if ann_infos.endswith('\n'):
        ann_infos = ann_infos[:-2]
    ann_infos = ann_infos.split('\n\n')

    bbox_dir = os.path.join(ann_path, ann_image_name)
    with open(bbox_dir, "r") as f:
        bbox_infos = f.readlines()
    image_dir = os.path.join(image_path, ann_image_name.replace(".txt", ".jpg"))
    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_size = image.shape[:2]

    pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    color_pattern = r'\b(\w+),\s+(\w+):\s+\[([\d.]+),\s*([\d.]+)\]'


    index = 0
    for ann_info in ann_infos:
        class_line = []
        image_vis1 = image.copy()
        image_vis2 = image.copy()

        matches = re.findall(pattern, ann_info)
        for match in matches:
            x, y = match
            x = float(x)
            y = float(y)
            cv2.circle(image_vis1, (int(x*image.shape[1]), int(y*image.shape[0])), 2, (255, 0, 0), 4)
            cv2.putText(image_vis1, str(x), (int(x*image.shape[1]), int(y*image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            class_line = [x, y]
        
        class_name = ""
        for i in range(len(bbox_infos)):
            bbox_info = bbox_infos[i].split('\n')[0]
            if str(class_line) in bbox_info:
                class_name = bbox_info.split(':')[0].split(',')[0]
  
        for i in range(len(bbox_infos)):
            bbox_info = bbox_infos[i].split('\n')[0]
            if class_name in bbox_info:
                matches = re.findall(color_pattern, bbox_info)
                cv2.circle(image_vis2, (int(float(matches[0][2])*image.shape[1]), int(float(matches[0][3])*image.shape[0])), 2, (0, 255, 0), 4)
                x = bbox_info.split('[')[1].split(',')[0] + matches[0][1][0:3]
                cv2.putText(image_vis2, str(x),(int(float(matches[0][2])*image.shape[1]), int(float(matches[0][3])*image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        image_show = cv2.hconcat([image_vis1, image_vis2])

        subtitle_height = 150 
        subtitle_width = image_show.shape[1]
        subtitle_area = np.zeros((subtitle_height, subtitle_width, 3), dtype=np.uint8)
        subtitle_text_caption = ann_info.split('\n')[0]
        subtitle_text_annotation = ann_info.split('\n')[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(subtitle_text_caption, font, font_scale, font_thickness)[0]
        text_y = (subtitle_area.shape[0] + text_size[1]) // 2 -10

        position = (20, 20)
        final_image_with_text, y = put_text_with_wrap(subtitle_area, subtitle_text_caption, position)
        position = (position[0], y + 10)
        final_image_with_text, y = put_text_with_wrap(final_image_with_text, subtitle_text_annotation, position)






        final_image = np.vstack([image_show, subtitle_area])

        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

        save_image_name = os.path.join(save_path, ann_image_name.split('.')[0] + "_" + str(index) + "_.jpg")
        cv2.imwrite(save_image_name, final_image)
        index += 1 

    print(check_states)
    if sum(check_states) != 0:
        ann_image_dir = os.path.join(image_reannotation_annotation, ann_image_name)
        with open(ann_image_dir, "r") as f:
            ann_infos = f.read()
        if ann_infos.endswith('\n'):
            ann_infos = ann_infos[:-2]  
        ann_infos = ann_infos.split('\n\n')
        reason_dir = os.path.join(image_annotation_check, ann_image_name)
        with open(reason_dir, "r") as f:
            check_infos = f.read()
        if check_infos.endswith('\n'):
            check_infos = check_infos[:-2]  
        check_infos = check_infos.split('\n\n')
        check_infos = [check_infos[i] for i in range(len(check_infos)) if check_states[i] == 1]


        index = 0
        for ann_info, check_info in zip(ann_infos, check_infos):
            class_line = []
            image_vis1 = image.copy()
            image_vis2 = image.copy()

            matches = re.findall(pattern, ann_info)
            for match in matches:
                x, y = match
                x = float(x)
                y = float(y)
                cv2.circle(image_vis1, (int(x*image.shape[1]), int(y*image.shape[0])), 2, (255, 0, 0), 4)
                cv2.putText(image_vis1, str(x), (int(x*image.shape[1]), int(y*image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                class_line = [x, y]
            
            class_name = ""
            for i in range(len(bbox_infos)):
                bbox_info = bbox_infos[i].split('\n')[0]
                if str(class_line) in bbox_info:
                    print("bbox_info", bbox_info)
                    class_name = bbox_info.split(':')[0]
            for i in range(len(bbox_infos)):
                bbox_info = bbox_infos[i].split('\n')[0]
                if class_name in bbox_info:
                    matches = re.findall(color_pattern, bbox_info)

                    cv2.circle(image_vis2, (int(float(matches[0][2])*image.shape[1]), int(float(matches[0][3])*image.shape[0])), 2, (0, 255, 0), 4)
                    x = bbox_info.split('[')[1].split(',')[0] + matches[0][1]
                    cv2.putText(image_vis2, str(x),(int(float(matches[0][2])*image.shape[1]), int(float(matches[0][3])*image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            image_show = cv2.hconcat([image_vis1, image_vis2])

            subtitle_height = 500 
            subtitle_width = image_show.shape[1]
            subtitle_area = np.zeros((subtitle_height, subtitle_width, 3), dtype=np.uint8)

            subtitle_text_caption = ann_info.split('\n')[0]
            subtitle_text_annotation = ann_info.split('\n')[1]
            subtitle_text_reason = check_info.split('\n')[-1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_size = cv2.getTextSize(subtitle_text_caption, font, font_scale, font_thickness)[0]
            text_y = (subtitle_area.shape[0] + text_size[1]) // 2 -10

            position = (20, 30)
            final_image_with_text, y = put_text_with_wrap(subtitle_area, subtitle_text_caption, position)

            position = (position[0], y + 10)
            final_image_with_text, y = put_text_with_wrap(final_image_with_text, subtitle_text_annotation, position)

            position = (position[0],  y + 10)
            final_image_with_text, y = put_text_with_wrap(final_image_with_text, subtitle_text_reason, position)



            final_image = np.vstack([image_show, subtitle_area])

            final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

            save_image_name = os.path.join(save_path, ann_image_name.split('.')[0] + "_" + str(index) + "_re.jpg")
            cv2.imwrite(save_image_name, final_image)
            index += 1 




def new_vis_image(ann_image_name):

    ann_image_dir = os.path.join(image_annotation_orginal, ann_image_name)
    with open(ann_image_dir, "r") as f: 
        ann_infos = f.read()
    if ann_infos.endswith('\n'):
        ann_infos = ann_infos[:-1]
    ann_infos = ann_infos.split('\n\n')

    bbox_dir = os.path.join(ann_path, ann_image_name)
    with open(bbox_dir, "r") as f:
        bbox_infos = f.readlines()
    image_dir = os.path.join(image_path, ann_image_name.replace(".txt", ".jpg"))
    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_size = image.shape[:2]

    pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    color_pattern = r'\b(\w+),\s+(\w+):\s+\[([\d.]+),\s*([\d.]+)\]'

    index = 0
    for ann_info in ann_infos:
        class_line = []
        image_vis1 = image.copy()
        image_vis2 = image.copy()
        matches = re.findall(pattern, ann_info)
        for match in matches:
            x, y = match
            x = float(x)
            y = float(y)
            cv2.circle(image_vis1, (int(x*image.shape[1]), int(y*image.shape[0])), 2, (255, 0, 0), 4)
            cv2.putText(image_vis1, str(x), (int(x*image.shape[1]), int(y*image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            class_line = [x, y]
        
        class_name = ""
        for i in range(len(bbox_infos)):
            bbox_info = bbox_infos[i].split('\n')[0]
            if str(class_line) in bbox_info:
                class_name = bbox_info.split(':')[0].split(',')[0]
  
        for i in range(len(bbox_infos)):
            bbox_info = bbox_infos[i].split('\n')[0]
            if class_name in bbox_info:
                matches = re.findall(color_pattern, bbox_info)
                cv2.circle(image_vis2, (int(float(matches[0][2])*image.shape[1]), int(float(matches[0][3])*image.shape[0])), 2, (0, 255, 0), 4)
                x = bbox_info.split('[')[1].split(',')[0] + matches[0][1][0:3]
                cv2.putText(image_vis2, str(x),(int(float(matches[0][2])*image.shape[1]), int(float(matches[0][3])*image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        image_show = cv2.hconcat([image_vis1, image_vis2])

        subtitle_height = 150 
        subtitle_width = image_show.shape[1]
        subtitle_area = np.zeros((subtitle_height, subtitle_width, 3), dtype=np.uint8)

        subtitle_text_caption = ann_info.split('\n')[0]
        subtitle_text_annotation = ann_info.split('\n')[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(subtitle_text_caption, font, font_scale, font_thickness)[0]
        text_y = (subtitle_area.shape[0] + text_size[1]) // 2 -10

        position = (20, 20)
        final_image_with_text, y = put_text_with_wrap(subtitle_area, subtitle_text_caption, position)
        position = (position[0], y + 10)
        final_image_with_text, y = put_text_with_wrap(final_image_with_text, subtitle_text_annotation, position)






        final_image = np.vstack([image_show, subtitle_area])

        final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

        save_image_name = os.path.join(save_path, ann_image_name.split('.')[0] + "_" + str(index) + "_.jpg")
        cv2.imwrite(save_image_name, final_image)
        index += 1 



if __name__ == "__main__":
    for ann_image_name in ann_image_names:
        try:
            new_vis_image(ann_image_name)
        except:
            paths = os.path.join(image_annotation_orginal, ann_image_name)
            print("wrong", paths)
            continue
