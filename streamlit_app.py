"""
Main entry point for FireSafety ML application.
This file is used by Streamlit for deployment.
"""

# Include full application code for simplicity
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure application modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the app module and run main
try:
    import app
    app.main()
except Exception as e:
    st.error(f"Error: {e}")
    st.write("""
    # Fire Safety ML Application
    
    This application combines machine learning with fire safety education.
    """)
    
    # Display debug info
    st.info("Application could not be loaded. Please check dependencies and file structure.")
    st.write("Files available:")
    st.code("\n".join(os.listdir(current_dir)))