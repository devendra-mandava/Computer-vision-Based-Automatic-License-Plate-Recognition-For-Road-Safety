#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it (optional, for debugging)
set -x

# Step 1: Create and activate a Python virtual environment
# if [ ! -d "venv" ]; then
#     echo "Creating Python virtual environment..."
#     python3 -m venv venv
# fi

# echo "Activating virtual environment..."
# source venv/bin/activate

# Step 2: Install required dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Step 3: Set up required directories
echo "Setting up directories..."
mkdir -p uploads
mkdir -p static/output
mkdir -p data/output

# Step 4: Run the Flask application
echo "Running the Flask application..."
python app.py

# Step 5: Notify the user of completion
echo "Application is running. Visit http://127.0.0.1:5000 to access the application."
