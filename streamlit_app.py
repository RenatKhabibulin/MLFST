"""
Main entry point for the ML FireSafetyTutor Streamlit application.
This is a self-contained file for streamlit.io deployment.
"""

import streamlit as st
import os
import sys

# Add the current directory to the Python path if it's not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the app module function directly
try:
    from app import main as app_main
    
    # Run the main application function
    app_main()
    
except ImportError as e:
    st.error(f"Error importing application modules: {e}")
    st.write("""
    ## ML FireSafetyTutor Application
    
    This application provides interactive machine learning education focused on fire safety topics.
    
    Unfortunately, there was an error loading the application.
    
    ### Troubleshooting:
    - Check that all required Python packages are installed
    - Ensure the application files are correctly structured
    """)
    
    # Display the current directory structure for debugging
    st.write("### Current Directory Structure:")
    files = os.listdir(current_dir)
    st.write(", ".join(files))