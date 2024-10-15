import pymupdf
import glob
from PIL import Image
import io


for file in glob.glob("pdfs/provas/*.pdf"):
    doc = pymupdf.open(file)
    for idx_page, page in enumerate(doc):
        pixels = page.get_pixmap()
        page_image = pixels.tobytes()
        page_image = Image.open(io.BytesIO(page_image))
        name_file = file.split("\\")[-1].split(".")[0]
        page_image.save(f"imgs\\imgs_pages\\{name_file}_p_{idx_page}.png")
    

