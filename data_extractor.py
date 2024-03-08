import os

from extractor import Extractor

extractor = Extractor()

# source_dir = "../CBSL-data/"
# output_dir = "../Extracted-text-CBSL-data-new/"

source_dir = "."
output_dir = "."
count = 0
error_count = 0

error_file_path = "errored-files.txt"

if not os.path.exists(error_file_path):
    with open(error_file_path, 'w') as file:
        file.write("")

with open(error_file_path, 'r', encoding='utf-8') as error_file:
    error_filenames = error_file.read().split('\n')

for root, directories, files in os.walk(source_dir):
    for directory in directories:
        # Create corresponding directories in the output directory
        output_subdir = os.path.join(
            output_dir, os.path.relpath(root, source_dir), directory)
        os.makedirs(output_subdir, exist_ok=True)

    for file in files:
        if file.endswith('.xlsx'):
            try:
                input_file_path = os.path.join(root, file)

                # Determine the output file path
                output_file_path = os.path.join(
                    output_dir, os.path.relpath(root, source_dir), file[:file.rfind('.')+1] + 'txt')

                if not os.path.exists(output_file_path) and input_file_path not in error_filenames:
                    text = extractor.extractor(input_file_path)

                    # Write the extracted text to the output file
                    with open(output_file_path, 'w', encoding='utf-8') as write_file:
                        write_file.write(text)

                    count += 1
                    print(
                        f"{count} - {file} is extracted and saved to {output_file_path}")

            except Exception as e:
                error_count += 1
                print(f"{error_count}E - {e} - {input_file_path}")
                with open(error_file_path, 'a', encoding='utf-8') as error_file_write:
                    error_file_write.write(f"{e} - {input_file_path}\n")


print(f"Success: {count} - Errors: {error_count}")
