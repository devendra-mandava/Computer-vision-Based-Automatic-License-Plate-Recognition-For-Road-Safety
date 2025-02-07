ALPR Web Application
This project is a web-based Automatic License Plate Recognition (ALPR) system. It detects vehicles and license plates in images or videos and extracts text from the plates.

Features
License Plate Detection: Detect license plates using YOLO.
Text Recognition (OCR): Extract text using EasyOCR.
Web Interface: Simple upload and preview functionality.
Processed Results: View processed images/videos with bounding boxes and license plate text.
How to Run
Step 1: Clone the Repository
bash
Copy code
git clone https://github.com/your-repo-name/ALPR-WebApp.git
cd ALPR-WebApp
Step 2: Install Dependencies
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Step 3: Run the Application
Run the app using the run.sh script:

bash
Copy code
bash run.sh
Alternatively, run directly:

bash
Copy code
python app.py
Step 4: Open the App
Navigate to http://127.0.0.1:5000/ in your browser.

Usage
Upload Files:
Upload an image (.jpg, .jpeg, .png) or video (.mp4, .avi).
View Results:
Images: Processed image with bounding boxes and license plate text.
Videos: Processed video with bounding boxes and license plate text.
Expected Outputs
Image:

Bounding boxes around detected license plates.
Extracted text with confidence scores.
Video:

Annotated frames with bounding boxes and license plate text.
No Plates Detected:

Message displayed if no plates are detected.
run.sh Script
The run.sh script simplifies running the app:

bash
Copy code
#!/bin/bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
Make it executable:

bash
Copy code
chmod +x run.sh
./run.sh
Enjoy using the ALPR Web App! 🚗
