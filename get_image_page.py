import pymupdf
import glob
from PIL import Image
import io
import os
import pytesseract

doc = pymupdf.open("pdfs/manual.pdf")
page = doc[0]
page_image = page.get_pixmap()
page_image.tobytes()

image = Image.open(io.BytesIO(page_image.tobytes()))
tesseract_text = pytesseract.image_to_string(image, lang='por')


for file in glob.glob("pdfs/*.pdf"):
    doc = pymupdf.open(file)
    text=""
    for idx_page, page in enumerate(doc):
        pixels = page.get_pixmap()
        page_image = pixels.tobytes()
        page_image = Image.open(io.BytesIO(page_image))
        text_extract = pytesseract.image_to_string(page_image, lang='por')
        text += text_extract
        name_file = file.split("\\")[-1].split(".")[0]
        with open(f"txts\\{name_file}.txt", "w", encoding='utf-8') as f:
            f.write(text)





