#!/bin/bash

# Скрипт для запуска Streamlit в Streamlit Cloud

# Установка правильного порта для Streamlit Cloud
export PORT=8501

# Информация для отладки
echo "Starting Streamlit on port $PORT"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Files in current directory: $(ls -la)"

# Запуск Streamlit на нужном порту
streamlit run app_cloud.py --server.port $PORT