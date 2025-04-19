import fitz  # PyMuPDF
import requests
import json
from PIL import Image
import pytesseract
import io
import os
from flask import Flask, request, jsonify

# Groq API Setup
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Initialize Flask app
app = Flask(__name__)

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print("Error reading PDF:", e)
    return text.strip()

# Step 2: Extract text from Image using OCR
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print("Error reading image:", e)
    return ""

# Step 3: Analyze using Groq
def analyze_medical_report_with_groq(text):
    if not text:
        return "No text extracted from the file."

    prompt = f"""
You are a medical assistant AI. A patient has uploaded the following medical report:

{text}

Instructions:
1. Identify all test values (e.g., Hemoglobin, WBC, etc.).
2. Compare each with normal reference ranges.
3. Mention if each value is Normal / Low / High.
4. Summarize what these results could indicate (like possible conditions).
5. Suggest if a doctor consultation is needed and what questions the user should ask.

Provide a clear and simple explanation suitable for non-medical users.
"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        completion = response.json()
        return completion['choices'][0]['message']['content']
    except Exception as e:
        return f"Error during Groq API call:\n{e}"

# Step 4: Parse AI Output and format as JSON
def parse_and_format_result(output):
    result_data = {}
    
    # Initialize categories
    result_data["test_results"] = []
    result_data["summary"] = ""
    result_data["consultation_suggestion"] = ""

    lines = output.splitlines()
    
    # Extracting Test Values, Status, and Formatting them
    for line in lines:
        if line.strip():
            parts = line.split(":")
            if len(parts) > 1:
                test_name = parts[0].strip()
                status = parts[1].strip()

                # Creating the structured output for each test result
                result_data["test_results"].append({
                    "test_name": test_name,
                    "status": status
                })
    
    # Extract Summary and Consultation Suggestion
    summary_started = False
    for line in lines:
        if "Summary:" in line:
            summary_started = True
        if summary_started:
            if "Consultation Suggestion:" in line:
                result_data["consultation_suggestion"] = line.split("Consultation Suggestion:")[-1].strip()
            elif "Summary" in line and result_data["summary"] == "":
                result_data["summary"] = line.strip()

    return result_data

# Step 5: Handle File Upload and Processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save file temporarily to process it
    file_extension = os.path.splitext(file.filename)[1].lower()
    temp_file_path = f"temp_file{file_extension}"
    file.save(temp_file_path)

    # Check file type and extract text
    if file_extension == ".pdf":
        print("Reading the medical report from PDF...")
        file_text = extract_text_from_pdf(temp_file_path)
    elif file_extension in [".jpg", ".jpeg", ".png"]:
        print("Reading the medical report from Image...")
        file_text = extract_text_from_image(temp_file_path)
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    if not file_text:
        return jsonify({"error": "No text extracted from the file"}), 400

    print("Analyzing with Groq LLaMA model...\n")
    result = analyze_medical_report_with_groq(file_text)

    # Format the result into structured JSON
    structured_result = parse_and_format_result(result)

    # Clean up the temporary file
    os.remove(temp_file_path)

    return jsonify(structured_result)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
