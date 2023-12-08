import os

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

directory_path = r'C:\Users\dmpoo\OneDrive\Desktop\realvsfake_merged_cropped\real_cropped'
file_count = count_files(directory_path)
print(f"Number of files in the directory: {file_count}")
