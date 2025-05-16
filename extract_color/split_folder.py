import os
import shutil

def split_folder(folder_path, num_subfolders, save_path):
    # Create subfolders
    for i in range(num_subfolders):
        subfolder_path = os.path.join(save_path, f"subfolder_{i+1}")
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path, exist_ok=True)

    # Get list of files in the folder
    files = os.listdir(folder_path)

    # Calculate number of files per subfolder
    files_per_subfolder = len(files) // num_subfolders

    # Copy files to subfolders
    for i, file in enumerate(files):
        subfolder_index = i // files_per_subfolder
        subfolder_path = os.path.join(save_path, f"subfolder_{subfolder_index+1}")
        file_path = os.path.join(folder_path, file)
        shutil.copy(file_path, subfolder_path)

# Usage example
folder_path = "/data/sunzc/VCoR/my_train/green"
save_path = "/data/sunzc/VCoR/my_train/green_split"
num_subfolders = 10
split_folder(folder_path, num_subfolders, save_path)