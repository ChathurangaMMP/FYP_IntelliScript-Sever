import os
import shutil

source_dir = "../Extracted-text-CBSL-data-new/"
output_dir = "../Extracted-text-CBSL-data-new-cleaned/"
count = 0
error_count = 0


for root, directories, files in os.walk(source_dir):
    for directory in directories:
        # Create corresponding directories in the output directory
        output_subdir = os.path.join(
            output_dir, os.path.relpath(root, source_dir), directory)
        os.makedirs(output_subdir, exist_ok=True)

    for file in files:
        try:
            input_file_path = os.path.join(root, file)

            with open(input_file_path, 'r') as infile:
                data = infile.read()

            # Determine the output file path
            output_file_path = os.path.join(
                output_dir, os.path.relpath(root, source_dir), file[:-3] + 'txt')

            shutil.copy(input_file_path, output_file_path)

            count += 1

        except:
            error_count += 1

for root, dirs, files in os.walk(output_dir, topdown=False):
    for dir in dirs:
        folder_path = os.path.join(root, dir)
        if not os.listdir(folder_path):  # Check if folder is empty
            os.rmdir(folder_path)
            print(f"Deleted empty folder: {folder_path}")

print(f"Success: {count} - Errors: {error_count}")
