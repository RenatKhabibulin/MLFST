#!/bin/bash

# Скрипт для запуска Streamlit в Streamlit Cloud

# Установка правильного порта для Streamlit Cloud
export PORT=8501

# Информация для отладки
echo "Starting Streamlit on port $PORT"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"
echo "Files in current directory: $(ls -la)"
echo "Config directory content:"
ls -la .streamlit/

# Проверяем содержимое конфиг-файла
echo "Current config.toml:"
cat .streamlit/config.toml

# Перестраховка: создаем правильный конфиг
mkdir -p .streamlit
cat > .streamlit/config.toml << EOL
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
EOL

echo "Updated config.toml:"
cat .streamlit/config.toml

# Запуск Streamlit на нужном порту с явным указанием хоста
streamlit run app_cloud.py --server.port $PORT --server.address 0.0.0.0