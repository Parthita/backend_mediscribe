import fitz  
import requests
import json
from PIL import Image
import pytesseract
import os
from flask import Flask, request, jsonify

# Groq API Setup
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Initialize Flask app
app = Flask(__name__)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print("Error reading PDF:", e)
    return text.strip()

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print("Error reading image:", e)
    return ""

def analyze_medical_report_with_groq(text):
    if not text:
        return "No text extracted from the file."

    prompt = f"""
You are a medical assistant AI. A patient has uploaded the following medical report:

{text}

Instructions:
1. Identify all test values (e.g., Hemoglobin, WBC, etc.).
2. Provide historical values if available.
3. Compare each with normal reference ranges.
4. Mention if each value is Normal / Low / High.
5. Detect trend if multiple values are available (e.g., increasing/decreasing).
6. Provide a simple summary per test.
7. Finally, provide a doctor consultation section with:
   - Whether consultation is needed (true/false)
   - Reason why
   - Suggested questions to ask the doctor
Format your output in structured JSON like this:

{{
  "labHistory": {{
    "testName": [
      {{
        "value": <number>,
        "normalRange": [min, max],
        "status": "low/high/normal"
      }}
    ]
  }},
  "summary": {{
    "testName": {{
      "latestValue": <number>,
      "trend": "increasing/decreasing/stable",
      "status": "low/high/normal"
    }}
  }},
  "doctorConsultation": {{
    "recommended": true/false,
    "reason": "text",
    "questions": ["question 1", "question 2"]
  }}
}}
Only return valid JSON with no explanation.
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

def parse_and_format_result(output):
    try:
        return json.loads(output)
    except Exception as e:
        return {
            "error": "Failed to parse AI output as JSON",
            "raw_output": output,
            "details": str(e)
        }

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_extension = os.path.splitext(file.filename)[1].lower()
    temp_file_path = f"temp_file{file_extension}"
    file.save(temp_file_path)

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
    structured_result = parse_and_format_result(result)

    os.remove(temp_file_path)

    if isinstance(structured_result, str):
        return jsonify({"error": "Unexpected string output", "raw_output": structured_result}), 500

    return jsonify(structured_result)

if __name__ == "__main__":
    app.run(debug=True)
