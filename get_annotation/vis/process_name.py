import os

def remove_copy_suffix_from_filenames(directory):
    # 遍历指定目录中的所有文件和子目录
    for filename in os.listdir(directory):
        # 构造文件的完整路径
        file_path = os.path.join(directory, filename)
        # 检查是否是文件（而不是目录）
        if os.path.isfile(file_path):
            # 检查文件名是否包含 "- 副本"
            if " - 副本" in filename:
                # 移除 "- 副本" 部分
                new_filename = filename.replace(" - 副本", "")
                # 构造新文件名的完整路径
                new_file_path = os.path.join(directory, new_filename)
                # 重命名文件
                os.rename(file_path, new_file_path)
                print(f"Renamed '{filename}' to '{new_filename}'")

# 使用你的文件夹路径替换下面的'your_directory_path'
remove_copy_suffix_from_filenames('/home/sunzc/VisDroneAnnotation/train_checked/color/test_color_annotation_others'
)