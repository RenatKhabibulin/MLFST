#!/bin/bash

# Скрипт для подготовки проекта к деплою на Streamlit Cloud
# Ультралегкая версия для минимизации использования ресурсов

echo "Подготовка проекта к деплою на Streamlit Cloud (УЛЬТРАЛЕГКАЯ ВЕРСИЯ)..."

# 1. Копируем config_cloud.toml в config.toml
echo "1. Настройка конфигурации для порта 8501..."
cp .streamlit/config_cloud.toml .streamlit/config.toml
echo "   Конфигурация скопирована."

# 2. Используем минимальные зависимости
echo "2. Настройка минимальных зависимостей..."
if [ -f "requirements-ultralight.txt" ]; then
    echo "   requirements-ultralight.txt найден."
    echo "   Копируем его в requirements.txt для облегченного деплоя..."
    cp requirements-ultralight.txt requirements.txt
    echo "   requirements.txt создан с минимальными зависимостями."
else
    echo "   ВНИМАНИЕ: requirements-ultralight.txt не найден!"
    echo "   Создаем минимальный файл зависимостей..."
    echo "streamlit>=1.28.0" > requirements.txt
    echo "   Минимальный requirements.txt создан."
fi

# 3. Проверяем наличие app_cloud_ultralight.py
echo "3. Проверка файла точки входа (ультралегкой версии)..."
if [ -f "app_cloud_ultralight.py" ]; then
    echo "   app_cloud_ultralight.py найден."
    echo "   entry.sh настроен на использование ультралегкой версии."
else
    echo "   ВНИМАНИЕ: app_cloud_ultralight.py не найден!"
    echo "   Это критическая ошибка, деплой не будет работать!"
    exit 1
fi

# 4. Проверяем и настраиваем entry.sh для ультралегкой версии
echo "4. Настройка скриптов запуска и мониторинга..."
if [ -f "entry.sh" ]; then
    echo "   entry.sh найден."
    # Устанавливаем права на выполнение
    chmod +x entry.sh
    echo "   Права на выполнение entry.sh установлены."
    
    # Проверяем что entry.sh использует ультралегкую версию
    if grep -q "app_cloud_ultralight.py" entry.sh; then
        echo "   entry.sh настроен правильно на ультралегкую версию."
    else
        echo "   ВНИМАНИЕ: entry.sh не использует ультралегкую версию!"
        echo "   Пожалуйста, убедитесь что entry.sh запускает app_cloud_ultralight.py"
    fi
else
    echo "   ВНИМАНИЕ: entry.sh не найден!"
    echo "   Это может вызвать проблемы с портами при деплое."
    exit 1
fi

if [ -f "healthcheck.py" ]; then
    echo "   healthcheck.py найден."
else
    echo "   ВНИМАНИЕ: healthcheck.py не найден!"
    echo "   Это может затруднить диагностику проблем с приложением."
fi

if [ -f "Procfile" ]; then
    echo "   Procfile найден."
    # Проверяем, что Procfile содержит healthcheck
    if grep -q "healthcheck:" Procfile; then
        echo "   Procfile содержит настройку healthcheck."
    else
        echo "   Обновляем Procfile для поддержки healthcheck..."
        echo "healthcheck: python healthcheck.py" >> Procfile
        echo "   Procfile обновлен."
    fi
else
    echo "   ВНИМАНИЕ: Procfile не найден!"
    echo "   Создаем Procfile с healthcheck..."
    echo "web: bash entry.sh" > Procfile
    echo "healthcheck: python healthcheck.py" >> Procfile
    echo "   Procfile создан."
fi

# 5. Проверяем/создаем runtime.txt для Python 3.11
echo "5. Настройка версии Python..."
if [ -f "runtime.txt" ]; then
    echo "   runtime.txt найден."
    # Проверяем, что runtime.txt содержит Python 3.11
    if grep -q "python-3.11" runtime.txt; then
        echo "   runtime.txt содержит правильную версию Python."
    else
        echo "   Обновляем runtime.txt для Python 3.11..."
        echo "python-3.11" > runtime.txt
        echo "   runtime.txt обновлен."
    fi
else
    echo "   Создаем runtime.txt для Python 3.11..."
    echo "python-3.11" > runtime.txt
    echo "   runtime.txt создан."
fi

echo ""
echo "✅ Готово! Проект подготовлен к деплою УЛЬТРАЛЕГКОЙ версии на Streamlit Cloud."
echo ""
echo "⚠️ ВАЖНО: Это ДЕМОНСТРАЦИОННАЯ версия без функциональности машинного обучения!"
echo "Полная версия требует больше ресурсов, чем доступно в бесплатном тире Streamlit Cloud."
echo ""
echo "📋 Настройки для Streamlit Cloud:"
echo "   - Main file path: app_cloud_ultralight.py"
echo "   - Python version: 3.11"
echo "   - Requirements file: requirements.txt (минимальные зависимости)"
echo ""
echo "⚙️ После деплоя можно вернуться к локальной разработке полной версии:"
echo "./restore_local_config.sh"