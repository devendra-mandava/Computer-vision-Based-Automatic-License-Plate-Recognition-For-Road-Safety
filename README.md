# Computer Vision Based Automatic Licence Plate Recognition for Road Safety

## Overview  
This project is a web-based **Automatic License Plate Recognition (ALPR)** system that detects vehicles and license plates in images or videos and extracts text from the plates using **computer vision and deep learning**.  

## Features  
- **License Plate Detection:** Uses **YOLO (You Only Look Once)** for accurate and fast plate detection.  
- **Optical Character Recognition (OCR):** Extracts text from detected plates using **EasyOCR**.  
- **Web Interface:** Provides a user-friendly interface for uploading and processing images/videos.  
- **Processed Results:** Displays bounding boxes and extracted text for detected license plates.  

## Project Structure  
```
├── models/            # Pre-trained YOLO and OCR models  
├── static/            # Static assets (CSS, JS, images)  
├── templates/         # HTML templates for the web UI  
├── uploads/           # Stores uploaded images and videos  
├── processed/         # Stores processed images and videos  
├── app.py             # Flask web application  
├── run.sh             # Shell script to run the application  
├── requirements.txt   # Required dependencies  
├── README.md          # Project documentation  
└── .gitignore         # Ignored files and folders  
```

## Prerequisites  
Ensure you have the following installed before running the project:
- **Python 3.8+**  
- **pip**  
- **Flask**  
- **YOLOv5 (pre-trained model)**  
- **EasyOCR**  
- **OpenCV**  

## Installation  

### Step 1: Clone the Repository  
```bash
git clone https://github.com/devendra-mandava/Computer-vision-Based-Automatic-License-Plate-Recognition-For-Road-Safety.git

```

### Step 2: Create and Activate a Virtual Environment  

For macOS/Linux:  
```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:  
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies  
```bash
pip install -r requirements.txt
```

## Running the Application  

### Option 1: Using `run.sh` Script  
```bash
bash run.sh
```

### Option 2: Run Manually  
```bash
python app.py
```

### Step 4: Open the Web App  
Once the app is running, open your browser and navigate to:  
```
http://127.0.0.1:5000/
```

## Usage  

### Upload Files  
- Supported formats: **Images (.jpg, .jpeg, .png), Videos (.mp4, .avi)**  

### View Processed Results  
- **Images:** Bounding boxes around detected plates with extracted text.  
- **Videos:** Processed video with annotated license plates and extracted text.  

### Error Handling  
- If no plates are detected, a message is displayed.  

## Expected Outputs  

### Image Processing  
- **Bounding boxes** around detected license plates.  
- **Extracted text** with confidence scores.  

### Video Processing  
- **Annotated frames** with bounding boxes and recognized text.  

### No Plates Detected  
- Displays: `"No license plates detected in the input file."`  

## `run.sh` Script  
The `run.sh` script simplifies running the application:  
```bash
#!/bin/bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```
Make it executable:  
```bash
chmod +x run.sh
./run.sh
```

## Future Enhancements  
- **Real-time ALPR using live video streams.**  
- **Support for multiple OCR engines (Tesseract, PaddleOCR).**  
- **Database integration to store recognized plates.**  
- **Deployment on cloud platforms (AWS, GCP, Azure).**  

## References  
- **[YOLO Object Detection - YOLO Paper](https://arxiv.org/abs/1506.02640)**  
- **[EasyOCR Documentation - EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)**  
- **[Flask Web Framework - Flask Docs](https://flask.palletsprojects.com/)**  

## License  
This project is licensed under the MIT License.  

🚗 **Enjoy using the ALPR Web App!** 🚀  
