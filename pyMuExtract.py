import pymupdf
import glob

for file in glob.glob("pdfs/*.pdf"):
    reader = pymupdf.open(file) 
    text =""
    for page in reader:
        text+=page.get_text()
    # Save the text to a file
    name_file = file.split("\\")[-1].split(".")[0]
    with open(f"txts\\{name_file}.txt", "w", encoding='utf-8') as f:
        f.write(text)

print(f"Numero de paginas: {(reader.page_count)}")


