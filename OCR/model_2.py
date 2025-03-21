import pdfplumber
import pandas as pd
import re

def extract_data_from_pdf(pdf_path, output_excel):
    columns = [
        "Approved Intake", "No. of Male Students", "No. of Female Students", "Total Students",
        "Within State", "Outside State", "Outside Country", "Economically Backward", "Socially Challenged",
        "Full Tuition Fee (Govt.)", "Full Tuition Fee (Institution Funds)", "Full Tuition Fee (Private)",
        "No Tuition Fee Reimbursement", "First Year Intake", "First Year Admitted", "Lateral Entry",
        "Graduating in Time", "Placed Students", "Median Salary", "Higher Studies",
        "Ph.D Students", "Ph.D Graduated", "Annual Capital Expenditure", "Annual Operational Expenditure",
        "Patents Published", "Patents Granted", "Sponsored Projects", "Funding Agencies",
        "Total Amount Received", "Consultancy Projects", "Client Organizations",
        "Lifts/Ramps", "Special Toilets", "Wheelchairs/Transport"
    ]

    # Initialize data dictionary with empty lists
    data = {col: [] for col in columns}
    faculty_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Extract values using regex patterns
                extracted_values = {}
                for col, pattern in [
                    ("Approved Intake", r"Approved Intake\s*:\s*(\d+)"),
                    ("No. of Male Students", r"No. of Male Students\s*:\s*(\d+)"),
                    ("No. of Female Students", r"No. of Female Students\s*:\s*(\d+)"),
                    ("Total Students", r"Total Students\s*:\s*(\d+)"),
                    ("Within State", r"Within State.*?(\d+)"),
                    ("Outside State", r"Outside State.*?(\d+)"),
                    ("Outside Country", r"Outside Country.*?(\d+)"),
                    ("Economically Backward", r"Economically Backward.*?(\d+)"),
                    ("Socially Challenged", r"Socially Challenged.*?(\d+)"),
                    ("Full Tuition Fee (Govt.)", r"full tuition fee reimbursement from the State.*?(\d+)"),
                    ("Full Tuition Fee (Institution Funds)", r"full tuition fee reimbursement from Institution Funds.*?(\d+)"),
                    ("Full Tuition Fee (Private)", r"full tuition fee reimbursement from the Private Bodies.*?(\d+)"),
                    ("No Tuition Fee Reimbursement", r"who are not receiving full tuition fee reimbursement.*?(\d+)"),
                    ("First Year Intake", r"first year students intake.*?(\d+)"),
                    ("First Year Admitted", r"first year students admitted.*?(\d+)"),
                    ("Lateral Entry", r"Lateral entry.*?(\d+)"),
                    ("Graduating in Time", r"graduating in minimum stipulated time.*?(\d+)"),
                    ("Placed Students", r"students placed.*?(\d+)"),
                    ("Median Salary", r"Median salary.*?(\d+)"),
                    ("Higher Studies", r"students selected for Higher Studies.*?(\d+)"),
                    ("Ph.D Students", r"Ph\.D \(Student pursuing doctoral program.*?(\d+)"),
                    ("Ph.D Graduated", r"Ph\.D students graduated.*?(\d+)"),
                    ("Annual Capital Expenditure", r"Annual Capital Expenditure.*?(\d+)"),
                    ("Annual Operational Expenditure", r"Annual Operational Expenditure.*?(\d+)"),
                    ("Patents Published", r"Patents Published.*?(\d+)"),
                    ("Patents Granted", r"Patents Granted.*?(\d+)"),
                    ("Sponsored Projects", r"Sponsored Projects.*?(\d+)"),
                    ("Funding Agencies", r"Funding Agencies.*?(\d+)"),
                    ("Total Amount Received", r"Total Amount Received.*?(\d+)"),
                    ("Consultancy Projects", r"Consultancy Projects.*?(\d+)"),
                    ("Client Organizations", r"Client Organizations.*?(\d+)")
                ]:
                    match = re.search(pattern, text, re.DOTALL)
                    extracted_values[col] = match.group(1) if match else "N/A"

                # Extract PCS Facilities
                extracted_values["Lifts/Ramps"] = "Yes" if "Lifts/Ramps" in text else "No"
                extracted_values["Special Toilets"] = "Yes" if "specially designed toilets" in text else "No"
                extracted_values["Wheelchairs/Transport"] = "Yes" if "walking aids" in text else "No"

                # Append extracted values to their respective lists
                for col in columns:
                    data[col].append(extracted_values.get(col, "N/A"))

                # Extract Faculty Details
                faculty_section = re.findall(
                    r"(\d+)\s+([\w\s]+)\s+(\d+)\s+([\w\s]+)\s+(\w+)\s+([\w\s]+)\s+(\d+)\s+(Yes|No)\s+(\d{4}-\d{2}-\d{2})\s+([\w\s]*)\s+([\w\s]*)", 
                    text
                )
                for faculty in faculty_section:
                    faculty_data.append(list(faculty))

    # Convert extracted data to Pandas DataFrame
    df = pd.DataFrame(data)

    faculty_df = pd.DataFrame(
        faculty_data, 
        columns=["Srno", "Name", "Age", "Designation", "Gender", "Qualification", 
                 "Experience", "Currently Working", "Joining Date", "Leaving Date", "Association Type"]
    )

    # Save to Excel
    with pd.ExcelWriter(output_excel) as writer:
        df.to_excel(writer, sheet_name="Institution Data", index=False)
        faculty_df.to_excel(writer, sheet_name="Faculty Details", index=False)

    print(f"Extracted data saved to {output_excel}")

# Run the script
pdf_path = "E:\\Recommendation_system\\Other stuffs\\NIRF Engineering 2025.pdf"
output_excel = "output_2.xlsx"
extract_data_from_pdf(pdf_path, output_excel)
