import os

from extractor import Extractor


source_dir = "../CBSL-data/"
output_dir = "../Extracted-text-CBSL-data-new/"
count = 0
error_count = 0

with open('errored-files.txt', 'r', encoding='utf-8') as error_file:
    error_filenames = error_file.read().split('\n')

for root, directories, files in os.walk(source_dir):
    for directory in directories:
        # Create corresponding directories in the output directory
        output_subdir = os.path.join(
            output_dir, os.path.relpath(root, source_dir), directory)
        os.makedirs(output_subdir, exist_ok=True)

    for file in files:
        try:
            input_file_path = os.path.join(root, file)

            # Determine the output file path
            output_file_path = os.path.join(
                output_dir, os.path.relpath(root, source_dir), file[:-3] + 'txt')

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
            with open('errored-files.txt', 'a', encoding='utf-8') as error_file_write:
                error_file_write.write(input_file_path+'\n')
