"""
ML FireSafetyTutor application main file
"""

# This module is a placeholder to ensure the package structure is complete
# The actual app functionality is in the root app.py file

# Import necessary modules from the parent application
import sys
import os

# Get the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the Python path if it's not already there
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the main app module's functions
from app import main as app_main

# Define a wrapper function that calls the main app function
def main():
    """Run the main application"""
    app_main()