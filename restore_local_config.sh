#!/bin/bash

# Скрипт для возврата к локальной конфигурации после подготовки к деплою

echo "Восстановление локальной конфигурации..."

# 1. Восстанавливаем оригинальный config.toml
if [ -f ".streamlit/config.toml.backup" ]; then
    echo "1. Восстановление конфигурации для порта 5000..."
    cp .streamlit/config.toml.backup .streamlit/config.toml
    echo "   Конфигурация восстановлена."
else
    echo "1. ОШИБКА: Файл резервной копии .streamlit/config.toml.backup не найден!"
    echo "   Создаем конфигурацию для порта 5000 вручную..."
    
    # Создаем конфигурацию для порта 5000
    cat > .streamlit/config.toml << 'EOL'
[server]
headless = true
address = "0.0.0.0"
# Для локального запуска используем порт 5000
# Для Streamlit Cloud следует использовать 8501
port = 5000
enableCORS = false
enableXsrfProtection = false
EOL
    
    echo "   Конфигурация для локального запуска создана."
fi

echo ""
echo "Готово! Конфигурация восстановлена для локальной разработки."
echo "Приложение теперь будет запускаться на порту 5000."
echo ""
echo "Чтобы запустить приложение, используйте команду:"
echo "streamlit run app.py --server.port 5000"