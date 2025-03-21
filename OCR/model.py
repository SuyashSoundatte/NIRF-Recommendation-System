import pdfplumber
import pandas as pd
import pytesseract
from PIL import Image
import pdf2image

def extract_tables_from_pdf(pdf_path, output_excel):
    data_frames = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table)
                data_frames.append(df)

    if data_frames:
        final_df = pd.concat(data_frames, ignore_index=True)
        final_df.to_excel(output_excel, index=False)
        print(f"Data extracted and saved to {output_excel}")
    else:
        print("No tables found in the PDF.")

def extract_text_from_scanned_pdf(pdf_path, output_excel):
    images = pdf2image.convert_from_path(pdf_path)
    extracted_text = []

    for img in images:
        text = pytesseract.image_to_string(img)
        extracted_text.append(text)
    
    df = pd.DataFrame({'Extracted Text': extracted_text})
    df.to_excel(output_excel, index=False)
    print(f"Extracted text saved to {output_excel}")


pdf_path = "E:\\Recommendation_system\\Other stuffs\\NIRF Engineering 2025.pdf"
output_excel = "output.xlsx"


extract_tables_from_pdf(pdf_path, output_excel)


