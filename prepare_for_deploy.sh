#!/bin/bash

# Скрипт для подготовки проекта к деплою на Streamlit Cloud

echo "Подготовка проекта к деплою на Streamlit Cloud..."

# 1. Копируем config_cloud.toml в config.toml
echo "1. Настройка конфигурации для порта 8501..."
cp .streamlit/config_cloud.toml .streamlit/config.toml
echo "   Конфигурация скопирована."

# 2. Проверяем наличие requirements-clean.txt
echo "2. Проверка файла зависимостей..."
if [ -f "requirements-clean.txt" ]; then
    echo "   requirements-clean.txt найден."
    echo "   Вы можете скопировать его в requirements.txt при деплое:"
    echo "   cp requirements-clean.txt requirements.txt"
else
    echo "   ВНИМАНИЕ: requirements-clean.txt не найден!"
    echo "   Создайте этот файл с совместимыми версиями пакетов."
fi

# 3. Проверяем наличие app_cloud.py
echo "3. Проверка файла точки входа..."
if [ -f "app_cloud.py" ]; then
    echo "   app_cloud.py найден."
    echo "   Используйте его как Main file path в настройках Streamlit Cloud."
else
    echo "   ВНИМАНИЕ: app_cloud.py не найден!"
    echo "   Создайте этот файл для деплоя в Streamlit Cloud."
fi

# 4. Проверяем наличие entry.sh и Procfile
echo "4. Проверка скрипта запуска..."
if [ -f "entry.sh" ]; then
    echo "   entry.sh найден."
    if [ -f "Procfile" ]; then
        echo "   Procfile найден."
    else
        echo "   ВНИМАНИЕ: Procfile не найден!"
        echo "   Создаем Procfile..."
        echo "web: bash entry.sh" > Procfile
        echo "   Procfile создан."
    fi
    # Устанавливаем права на выполнение
    chmod +x entry.sh
    echo "   Права на выполнение entry.sh установлены."
else
    echo "   ВНИМАНИЕ: entry.sh не найден!"
    echo "   Это может вызвать проблемы с портами при деплое."
fi

# 5. Напоминаем о Python 3.11
echo "5. Напоминание о версии Python..."
echo "   Рекомендуется использовать Python 3.11 (не 3.12) для деплоя."

echo ""
echo "Готово! Проект подготовлен к деплою на Streamlit Cloud."
echo "Примечание: Не забудьте указать следующие настройки в Streamlit Cloud:"
echo "   - Main file path: app_cloud.py"
echo "   - Python version: 3.11"
echo "   - Requirements file: requirements-clean.txt или requirements.txt"
echo ""
echo "Для возврата к локальной разработке, восстановите оригинальный config.toml:"
echo "./restore_local_config.sh"