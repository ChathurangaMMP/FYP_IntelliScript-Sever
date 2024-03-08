from io import BytesIO
from img2table.document import PDF

import os
import fitz
import pytesseract
from pdf2image import convert_from_path
from img2table.ocr import TesseractOCR
from img2table.document import pdf
from tabulate import tabulate
import json
import pandas as pd
import re
import fitz  # PyMuPDF
from tabulate import tabulate
import csv
import whisper
from fpdf import FPDF

from spire.doc import FileFormat as FPdoc
from spire.doc import *
from spire.doc.common import *

from spire.presentation import *
from spire.presentation.common import *


import layoutparser as lp
import cv2
import pytesseract
from PIL import Image

from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection
import torch


class Extractor:
    def __init__(self):
        self.create_folder("temp")
        # Load the LayoutParser model
        self.layout_model = lp.PaddleDetectionLayoutModel(
            config_path="lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet/config",
            threshold=0.5,
            label_map={0: "Text", 1: "Title",
                       2: "List", 3: "Table", 4: "Figure"},
            enforce_cpu=False,
            enable_mkldnn=True
        )

        self.feature_extractor = DetrFeatureExtractor()
        self.structure_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition")
        self.label_dict = self.structure_model.config.id2label

        # Create a Presentation object
        self.presentation = Presentation()
        # Create a Document object
        self.document = Document()

    # service function

    def extractor(self, path):
        extension = path.split("/")[-1].split(".")[-1]

        if (extension in ["jpg", "png", "jpeg"]):
            return self.image_extractor(path)
            # return self.doc_extractor(path)

        elif (extension in ["docx", "doc"]):
            return self.doc_extractor(path)

        elif (extension in ["pptx", "ppt"]):
            return self.pptx_extractor(path)

        elif (extension in ["wav", "mp3"]):
            return self.audio_extractor(path)

        elif (extension in ["csv"]):
            return self.csv_extractor(path)

        elif (extension in ["xlsx"]):
            return self.xlsx_extractor(path)

        elif (extension in ["xls"]):
            return self.xls_extractor(path)

        elif (extension in ["pdf"]):
            return self.pdf_extractor(path)

        else:
            return None

    # Utilz

    def create_folder(self, folder_path):
        try:
            os.makedirs(folder_path)
            print(f"Folder created at {folder_path}")
        except FileExistsError:
            print(f"Folder already exists at {folder_path}")

    def delete_folder(self, folder_path):
        try:
            os.rmdir(folder_path)
            print(f"Folder deleted at {folder_path}")
        except FileNotFoundError:
            print(f"Folder not found at {folder_path}")

    def delete_file(self, file_path):

        try:
            os.remove(file_path)
            print(f"File deleted at {file_path}")
        except FileNotFoundError:
            print(f"File not found at {file_path}")

    # Image extractor

    def table_extractor(self, pil_image, structure_model, feature_extractor, label_dict):
        width, height = pil_image.size
        pil_image.resize((int(width*0.5), int(height*0.5)))

        encoding = feature_extractor(pil_image, return_tensors="pt")
        encoding.keys()

        model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition")

        with torch.no_grad():
            outputs = model(**encoding)

        target_sizes = [pil_image.size[::-1]]
        results = feature_extractor.post_process_object_detection(
            outputs, threshold=0.6, target_sizes=target_sizes)[0]

        labels, boxes = results['labels'], results['boxes']

        column_header = None
        table_rows = []
        table_column = []
        for label, (xmin, ymin, xmax, ymax) in zip(labels.tolist(), boxes.tolist()):
            label = label_dict[label]
            print(label)
            if label in ['table row', 'table column header', 'table column']:
                if label == "table column header":
                    column_header = [ymin, ymax]
                elif label == "table column":
                    table_column.append([xmin, xmax])
                else:
                    table_rows.append([ymin, ymax])

        table_column_header = []
        table_cell = []

        if (column_header != None):
            for i in table_column:
                cropped_image = pil_image.crop(
                    (i[0], column_header[0], i[1], column_header[1]))
                text = pytesseract.image_to_string(cropped_image, lang='eng')
                print(text)
                table_column_header.append(text)

        for j in table_rows:
            table_cell.append([])
            for i in table_column:
                cropped_image = pil_image.crop((i[0], j[0], i[1], j[1]))
                text = pytesseract.image_to_string(cropped_image, lang='eng')

                table_cell[-1].append(text)

        data = []
        data.append(table_column_header)
        data = data+table_cell

        return tabulate(data, headers="firstrow", tablefmt="grid")

    def image_extractor(self, image_file):
        image = cv2.imread(image_file)
        image = image[..., ::-1]

        # Detect the layout of the image
        layout = self.layout_model.detect(image)

        text = ""

        ind = 0
        for l in layout:
            if l.type == "Text":
                x1 = int(l.block.x_1)
                y1 = int(l.block.y_1)
                x2 = int(l.block.x_2)
                y2 = int(l.block.y_2)

                pil_image = Image.fromarray(cv2.cvtColor(
                    image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                text += pytesseract.image_to_string(
                    pil_image, lang='eng') + "\n"

            elif l.type == "Table":
                x1 = int(l.block.x_1)
                y1 = int(l.block.y_1)
                x2 = int(l.block.x_2)
                y2 = int(l.block.y_2)

                pil_image = Image.fromarray(cv2.cvtColor(
                    image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                pil_image.save(str(ind)+"saved_image.jpg")
                ind += 1
                text += self.table_extractor(pil_image, self.structure_model,
                                             self.feature_extractor, self.label_dict)+"\n"
                # text+= pytesseract.image_to_string(pil_image, lang='eng') +"\n"

        return text

    # Xls extractor

    def convert_xls_to_csv(self, xls_file, csv_file):
        data = pd.read_excel(xls_file, engine='xlrd')
        data.to_csv(csv_file, index=False)

    def xls_extractor(self, xls_path):
        name = xls_path.split("\\")[-1].split(".")[0]+".csv"
        self.convert_xls_to_csv(xls_path, "temp\\"+name)
        text = self.csv_extractor("temp\\"+name)
        self.delete_file("temp\\"+name)
        return text

    # Xlsx extractor

    def convert_xlsx_to_csv(self, xlsx_file, csv_file):
        data = pd.read_excel(xlsx_file)
        data.to_csv(csv_file, index=False)

    def xlsx_extractor(self, xlsx_path):
        name = xlsx_path.split("\\")[-1].split(".")[0]+".csv"
        self.convert_xlsx_to_csv(xlsx_path, "temp\\"+name)
        text = self.csv_extractor("temp\\"+name)
        self.delete_file("temp\\"+name)
        return text

    # Doc extractor
    def convert_docx_to_pdf(self, docx_file, pdf_file):

        self.document.LoadFromFile(rf"{docx_file}")

        self.document.SaveToFile(pdf_file, FPdoc.PDF)
        self.document.Close()

    def doc_extractor(self, doc_path):
        name = doc_path.split("\\")[-1].split(".")[0]+".pdf"
        self.convert_docx_to_pdf(doc_path, "temp\\"+name)
        text = self.pdf_extractor("temp\\"+name)
        self.delete_file("temp\\"+name)
        return text

    # PPTX extractor
    def convert_pptx_to_pdf(self, pptx_path, pdf_file):

        self.presentation.LoadFromFile(pptx_path)

        # Convert the presentation to PDF format
        self.presentation.SaveToFile(pdf_file, FileFormat.PDF)
        self.presentation.Dispose()
        print("Convertion done")

    def pptx_extractor(self, pptx_path):
        name = pptx_path.split("\\")[-1].split(".")[0]+".pdf"
        self.convert_pptx_to_pdf(pptx_path, "temp\\"+name)
        text = self.pdf_extractor("temp\\"+name)
        self.delete_file("temp\\"+name)
        return text

    # Audio extractor
    def audio_extractor(self, audio_path):
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        lines = result['text'].splitlines()
        text = ""
        for line in lines:
            text += line+"\n"
            print(line)
        return text

    # CSV extractor
    def read_csv_and_format(self, csv_file):
        # Read CSV file
        with open(csv_file, 'r', encoding='ISO-8859-1') as file:
            reader = csv.reader(file)
            data = list(reader)

        # Format data using tabulate
        table = tabulate(data, headers="firstrow", tablefmt="grid")

        # Print formatted table
        return table

    def csv_extractor(self, csv_path):
        table = self.read_csv_and_format(csv_path)

        return table

    # Pdf extractor
    def pdf_page_to_json(self, doc, page_number):

        # Get the specified page
        page = doc.load_page(page_number - 1)

        # Extract the page text
        text = page.get_text('json')
        return text

    def replace_substring(self, original_string, start_index, new_substring):

        before_substring = original_string[:start_index]
        after_substring = original_string[start_index + len(new_substring):]
        new_string = before_substring + new_substring + after_substring
        return new_string

    def json_to_text(self, json_data, wid_fac):
        height = json_data['height']
        width = json_data['width']

        factor_hei = 75/height
        factor_wid = wid_fac

        page = []
        for hei in range(int(height*factor_hei)):
            line = {"starter": False, "fatcor": factor_wid,
                    "data": " "*int(width*factor_wid)}
            page.append(line)

        for block in json_data['blocks']:
            try:
                for line in block['lines']:
                    for span in line['spans']:
                        x1, y1 = span['origin']

                        y1 = y1*factor_hei
                        x1 = x1*factor_wid

                        page[int(y1)]["data"] = self.replace_substring(
                            page[int(y1)]["data"], int(x1), span['text'])
            except Exception as e:
                print("error", e)

        created_page = ""
        for line in page:
            created_page += line["data"]+"\n"

        return created_page

    def squ_ava_fac(self, json_data):
        fac = 0
        nu = 0
        for block in json_data['blocks']:
            try:
                for line in block['lines']:
                    for span in line['spans']:
                        bx1, by1, bx2, by2 = span['bbox']
                        fac += (len(span['text'])/(bx2-bx1))**2
                        nu += 1
            except Exception as e:
                print("error", e)

        ava = 0

        if (nu != 0):

            ava = fac/nu

        return ava

    def replace_from_table(self, json_data, cordinates, tables, n):

        for i in range(n):
            cordinate = cordinates[i]
            table = tables[i]
            delete_block = []

            ycordinate = []
            for block_ind in range(len(json_data['blocks'])):
                block = json_data['blocks'][block_ind]
                try:
                    for line in block['lines']:
                        for span in line['spans']:

                            x1, y1, x2, y2 = span['bbox']
                            x1, y1, x2, y2 = x1*2.77, y1*2.77, x2*2.77, y2*2.77
                            if (((cordinate[0]-10 <= x1 and x2 <= cordinate[2]+10)) and ((cordinate[1]-10 <= y1 and y2 <= cordinate[3]+10))):
                                ycordinate.append(y1/2.77)
                                delete_block.append(block_ind)

                except:
                    print("Error")

            delete_block = list(set(delete_block))
            delete_block.sort(reverse=True)

            for del_ind in delete_block:
                del json_data['blocks'][del_ind]

            table_lis = table.split("\n")
            ycordinate = list(set(ycordinate))
            ycordinate.sort()

            sum = 0
            for i in range(1, len(ycordinate)):
                sum += ycordinate[i] - ycordinate[i-1]
            space = sum/(len(ycordinate)-1)

            for row_table in range(len(table_lis)):

                new_block = {"lines": [{"spans": [{"origin": [
                    cordinate[0]/2.77, cordinate[1]/2.77+row_table*space], "text": table_lis[row_table]}]}]}
                json_data['blocks'].append(new_block)

        return json_data

    def tabulate_converter(self, json_data):
        # Convert JSON to a list of dictionaries

        if type(json_data) == list:
            table_data = [list(record.values()) for record in json_data]
            # Get headers from the first record
            headers = list(json_data[0].keys())

        elif type(json_data) == dict:
            table_data = list(json_data.values())

            # Get headers from the first record
            headers = list(json_data.keys())

        elif type(json_data) == str:

            # Convert the JSON string to a list of dictionaries
            json_data = json.loads(json_data)

            table_data = [list(record.values()) for record in json_data]
            # Get headers from the first record
            headers = list(json_data[0].keys())

        # Convert to Markdown table
        markdown_table = tabulate(table_data, headers, tablefmt="pipe")

        return markdown_table

    def table_titles(self, tables_dict):
        titles = []
        for t in tables_dict:
            if tables_dict[t] != []:
                t_index = 1
                for table in tables_dict[t]:
                    titles.append([(t+1, t_index), table.title])
                    t_index += 1

        return titles

    def pdf_extractor(self, pdf_path):

        doc = fitz.open(pdf_path)

        txt_content = """"""

        get_ava = 0
        ind = 0

        json_data_list = []

        for page_number in range(len(doc)):
            text = doc[page_number].get_text()
            if not text:

                page = doc.load_page(page_number)
                pix = page.get_pixmap()
                image = Image.frombytes(
                    "RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(image)
                txt_content += text + "\n"
                json_data = {"ocr": True, "text": txt_content}

            else:
                json_data = self.pdf_page_to_json(doc, page_number)
                json_data = json.loads(json_data)
                json_data["ocr"] = False
                json_data_list.append(json_data)
                ava = self.squ_ava_fac(json_data)
                if (ava != 0):
                    get_ava += ava
                    ind += 1

        try:
            wid_fac = (get_ava/ind)**0.5

        except:
            wid_fac = 0

        print(wid_fac)

        for json_data in json_data_list:
            if (json_data["ocr"]):
                txt = json_data["text"]
            else:
                txt = self.json_to_text(json_data, wid_fac)

            txt_content += txt

        return txt_content
