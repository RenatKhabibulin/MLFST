"""
Simplified app for Streamlit Cloud deployment
"""
import streamlit as st
import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Получение порта из переменной окружения или использование порта по умолчанию
PORT = int(os.environ.get("PORT", 8501))

st.title("ML FireSafety Tutor")
st.markdown("""
## Welcome to the ML FireSafety Tutor

This application provides interactive machine learning education focused on fire safety topics.

To use the full application, please run it locally with:
```
streamlit run app.py
```

### Why this simplified version?

Streamlit Cloud is having difficulty deploying the full application with its package structure.
For the complete application, please visit the GitHub repository and follow installation instructions.
""")

st.info("For the complete interactive ML FireSafety Tutor application, clone the repository and run locally.")

# Show key features list
st.subheader("Key Features")
st.markdown("""
- Interactive learning modules
- ML tutorials with fire safety applications
- Hands-on exercises and code samples
- Progress tracking
- Practical examples and data visualizations
""")

# Add a button that would direct to GitHub
if st.button("Visit GitHub Repository"):
    st.markdown("[GitHub Repository](https://github.com/yourusername/mlfiresafetytutor)")

# Explain deployment options
st.subheader("Deployment Options")
st.markdown("""
1. **Local Installation**: Clone the repository and run locally
2. **GitHub Pages**: For static content and documentation
3. **Custom Server**: Deploy to your own server for full functionality
""")