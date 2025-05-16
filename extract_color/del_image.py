# Utility script to delete files from folder_a that also exist in folder_b
import os

def delete_files_with_same_name(folder_a, folder_b):
    """
    Deletes files from folder_a that have matching filenames in folder_b
    
    Args:
        folder_a (str): Source directory containing files to potentially delete
        folder_b (str): Target directory to check for matching filenames
    """
    files_a = os.listdir(folder_a)
    files_b = os.listdir(folder_b)

    for file_a in files_a:
        if file_a in files_b:
            file_path = os.path.join(folder_a, file_a)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

# Example usage - modify these paths as needed
folder_a = "data/visdrone_color_image"  # Source directory
folder_b = "data/visdrone_tiny_cars_image"  # Target directory

if __name__ == '__main__':
    delete_files_with_same_name(folder_a, folder_b)