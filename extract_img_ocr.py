import pymupdf
import glob
from PIL import Image
import io
import os
import pytesseract

for file in glob.glob("pdfs/*.pdf"):
    doc = pymupdf.open(file)
    text = ""
    for idx_page, page in  enumerate(doc):
        text = page.get_text()
        if text:
            print(text)
            name_file = file.split("\\")[-1].split(".")[0]
            with open(f"txts\\{name_file}_p_{idx_page}.txt", "w", encoding='utf-8') as f:
                f.write(text)
        else:
            print("usando OCR")         
            images = page.get_images()
            if images:              
                for idx, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image = Image.open(io.BytesIO(image_bytes))
                    text += pytesseract.image_to_string(image, lang='por')
                    name_file = file.split("\\")[-1].split(".")[0]
                    with open(f"txts\\{name_file}_p_{idx_page}.txt", "w", encoding='utf-8') as f:
                        f.write(text)











for file in glob.glob("pdfs/*.pdf"):
    name_file = file.split("\\")[-1].split(".")[0]
    os.makedirs(f"imgs\\{name_file}")

              

