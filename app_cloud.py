"""
Simplified app for Streamlit Cloud deployment
"""
import streamlit as st
import os
import sys
import socket

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Получение порта из переменной окружения или использование порта по умолчанию
# Streamlit Cloud обычно использует порт 8501
PORT = int(os.environ.get("PORT", 8501))

# Расширенная информация о среде запуска (для отладки)
is_cloud = "STREAMLIT_CLOUD" in os.environ or "DEPLOY_URL" in os.environ
print(f"Running in {'Streamlit Cloud' if is_cloud else 'local environment'}")
print(f"Using port: {PORT}")
print(f"Python version: {sys.version}")
print(f"Streamlit version: {st.__version__}")
print(f"Environment variables: {dict(os.environ)}")

# Проверка доступности порта
def check_port(port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('0.0.0.0', port))
        s.close()
        return True
    except:
        return False

if check_port(PORT):
    print(f"Port {PORT} is available")
else:
    print(f"WARNING: Port {PORT} may already be in use")

# Настройка конфигурации Streamlit программно
st.set_page_config(
    page_title="ML FireSafety Tutor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Индикатор состояния сервера
with st.expander("Server Information (Debug)", expanded=False):
    st.write(f"Running on port: {PORT}")
    st.write(f"In cloud environment: {is_cloud}")
    st.write(f"Python version: {sys.version}")
    st.write(f"Streamlit version: {st.__version__}")
    st.write(f"Current directory: {current_dir}")
    st.write(f"Socket can bind to port 8501: {check_port(8501)}")
    st.write("Environment variables may be viewed in the server logs")

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