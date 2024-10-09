import pymupdf
import glob
from PIL import Image
import io
import os

for file in glob.glob("pdfs/*.pdf"):
    doc = pymupdf.open(file)
    name_file = file.split("\\")[-1].split(".")[0]
    os.makedirs(f"imgs\\{name_file}")
    for idx_page, page in  enumerate(doc):
        images = page.get_images()

        if images:
            for idx, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))
                image.save(f"imgs\\{name_file}\\image_p_{idx_page+1}_img_{idx+1}.{image_ext}")

              

