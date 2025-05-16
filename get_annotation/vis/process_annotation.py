import os
# Read txt file and:
# 1. Replace more than 3 consecutive \n with 2 \n
# 2. Ensure no \n at the beginning of text
def clean_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = content.lstrip('\n')

    cleaned_content = ''
    newline_count = 0

    for char in content:
        if char == '\n':
            newline_count += 1
        else:
            newline_count = 0

        if newline_count <= 2:
            cleaned_content += char


    while cleaned_content.endswith('\n\n\n'):
        cleaned_content = cleaned_content[:-1]

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
        
ann_dir = '/home/sunzc/VisDroneAnnotation/train_checked/color/annotation_2'
for ann_file in os.listdir(ann_dir):
    if ann_file.endswith('.txt'):
        clean_text_file(os.path.join(ann_dir, ann_file))
