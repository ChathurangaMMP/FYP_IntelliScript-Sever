import os
import fitz
import pytesseract
from pdf2image import convert_from_path
from img2table.document import PDF
from img2table.ocr import TesseractOCR
from tabulate import tabulate
import json
import pandas as pd
import re


def table_to_json_based_xlsx(xlsx_path, titles):
    xlsx_tables = []
    xlsx_file = pd.ExcelFile(xlsx_path)  # Replace with your file path
    sheet_names = xlsx_file.sheet_names

    for sheet_name in sheet_names:
        df = pd.read_excel(xlsx_file, sheet_name=sheet_name)

        match = re.search(r"Page (\d+) - Table (\d+)", sheet_name)
        page_number = ''
        table_number = ''
        if match:
            page_number, table_number = map(int, match.groups())

        title = ''
        if page_number != '' and table_number != '':
            title = [k[1]
                     for k in titles if k[0] == (page_number, table_number)][0]

        if df.empty:
            temp = [{}]
            for k in range(len(df.columns)):
                try:
                    temp[0][df.columns[k]] = ''
                except:
                    continue

            xlsx_tables.append(
                {'info': (page_number, table_number), 'title': title, 'data': json.dumps(temp)})
        else:
            xlsx_tables.append({'info': (page_number, table_number),
                               'title': title, 'data': df.to_json(orient="records")})

    return xlsx_tables


def table_extract(source):
    return source.extract_tables(ocr=ocr,
                                 implicit_rows=True,
                                 borderless_tables=True,
                                 min_confidence=90)


def table_titles(tables_dict):
    titles = []
    for t in tables_dict:
        if tables_dict[t] != []:
            t_index = 1
            for table in tables_dict[t]:
                titles.append([(t+1, t_index), table.title])
                t_index += 1

    return titles


def needs_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    for page in doc:
        # Extract text using PyMuPDF
        text = page.get_text()
        if not text:
            return True  # PDF contains no searchable text
    return False


def find_text(page_text, word):
    start_index = 0
    indexes = []
    word_count = page_text.count(word)
    for i in range(word_count):
        start_index = page_text.find(word, start_index)
        if start_index == -1:
            break

        end_index = start_index + len(word)

        indexes.append((start_index, end_index))
        start_index = end_index

    return indexes


def tabulate_converter(json_data):
    # Convert JSON to a list of dictionaries
    if type(json_data) == list:
        table_data = [list(record.values()) for record in json_data]
        # Get headers from the first record
        headers = list(json_data[0].keys())

    elif type(json_data) == dict:
        table_data = list(json_data.values())

        # Get headers from the first record
        headers = list(json_data.keys())

    # Convert to Markdown table
    markdown_table = tabulate(table_data, headers, tablefmt="pipe")

    return markdown_table


def extract_data(filepath):
    if filepath.endswith((".pdf")):
        pdf = PDF(filepath,
                  detect_rotation=False,
                  pdf_text_extraction=True)

        pdf.to_xlsx(xlsx_path,
                    ocr=ocr,
                    implicit_rows=True,
                    borderless_tables=True,
                    min_confidence=90)

        extracted_tables = table_extract(pdf)
        titles = table_titles(extracted_tables)

        tables = table_to_json_based_xlsx(xlsx_path, titles)
        text_all = ''

        doc = convert_from_path(
            filepath, 750, thread_count=10, grayscale=True, use_pdftocairo=True, size=3000)

        for i in range(len(doc)):
            page = doc[i]
            # Process each page

            text = pytesseract.image_to_string(
                page, config=custom_config, lang='eng')

            page_tables = []
            for tb in tables:
                if tb['info'][0] == i+1:
                    page_tables.append(tb)

            if page_tables == []:
                if text.strip() not in text_all:
                    text_all += text

            else:
                for tbl in page_tables:
                    starting_words = []
                    ending_words = []
                    table_json = json.loads(tbl['data'])

                    if len(table_json) >= 2:
                        starting_words.extend(table_json[0].keys())
                        starting_words.extend(table_json[0].values())

                        ending_words.extend(
                            list(table_json[-1].values())[::-1])
                        ending_words.extend(
                            list(table_json[-2].values())[::-1])
                    else:
                        starting_words.extend(table_json[0].keys())
                        starting_words.extend(table_json[0].values())

                        ending_words.extend(
                            list(table_json[-1].values())[::-1])

                    start_crop = 0
                    end_crop = 0
                    for w in starting_words:
                        if w != None:
                            matches = find_text(text, str(w))
                            if len(matches) == 1:
                                start_crop = matches[0][0]
                                break
                            else:
                                continue
                        else:
                            continue

                    for w in ending_words:
                        if w != None:
                            matches = find_text(text, str(w))
                            if len(matches) == 1:
                                end_crop = matches[0][1]
                                break
                            else:
                                continue
                        else:
                            continue

                    new_text = text[:start_crop]+'\n\n' + \
                        tabulate_converter(table_json)+'\n\n'+text[end_crop:]

                    new_text = new_text.split('| About')[0]
                    new_text = new_text.split('| f')[0]
                    new_text = new_text.split('| Contact Us')[0]

                    # text=new_text

                    if new_text.strip() not in text_all:
                        text_all += new_text
                        text = new_text

                    tables.remove(tbl)

    return text_all


ocr = TesseractOCR(n_threads=1, lang="eng")
custom_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 1 -l eng'

xlsx_path = 'temp/tables.xlsx'

input_file_path = 'tabular-pdfs/course-catalogue.pdf'

with open('test-data.txt', 'w', encoding='utf-8') as outfile:
    outfile.write(extract_data(input_file_path))
