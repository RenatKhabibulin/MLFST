#!/bin/bash

# Скрипт для запуска Streamlit в Streamlit Cloud
# Ультралегкая версия для запуска на Streamlit Cloud без ML зависимостей

# Установка правильного порта для Streamlit Cloud
export PORT=8501

# Информация для отладки
echo "Starting Streamlit on port $PORT"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "System memory info:"
free -h
echo "Disk space:"
df -h .

# Проверка статуса портов
echo "Port status (8501):"
nc -zv localhost 8501 || echo "Port 8501 is available"

# Перестраховка: создаем правильный конфиг
mkdir -p .streamlit
cat > .streamlit/config.toml << EOL
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 1
maxMessageSize = 5
EOL

echo "Config created. Starting ultralight version..."

# Запуск ультралегкой версии приложения без ML зависимостей
streamlit run app_cloud_ultralight.py --server.port $PORT --server.address 0.0.0.0