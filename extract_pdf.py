import os
import glob
from PyPDF2 import PdfReader


for file in glob.glob("pdfs/*.pdf"):
    reader = PdfReader(file) 
    text =""
    for page in reader.pages:
        text= text + page.extract_text()
    print(text)
    # Save the text to a file
    name_file = file.split("\\")[-1].split(".")[0]
    with open(f"txts\\{name_file}.txt", "w", encoding='utf-8') as f:
        f.write(text)

print(f"Numero de paginas: {len(reader.pages)}")