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
    echo "   Копируем его в requirements.txt для деплоя..."
    cp requirements-clean.txt requirements.txt
    echo "   requirements.txt создан."
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

# 4. Проверяем наличие entry.sh, Procfile и healthcheck.py
echo "4. Проверка скриптов запуска и мониторинга..."
if [ -f "entry.sh" ]; then
    echo "   entry.sh найден."
    # Устанавливаем права на выполнение
    chmod +x entry.sh
    echo "   Права на выполнение entry.sh установлены."
else
    echo "   ВНИМАНИЕ: entry.sh не найден!"
    echo "   Это может вызвать проблемы с портами при деплое."
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
echo "Готово! Проект подготовлен к деплою на Streamlit Cloud."
echo ""
echo "ВАЖНЫЕ ЗАМЕЧАНИЯ:"
echo "1. Убедитесь, что приложение использует порт 8501 в Streamlit Cloud"
echo "2. В случае ошибок при проверке состояния посмотрите project_cleanup/port_and_healthcheck.md"
echo "3. Если deploy не удался, проверьте логи в Streamlit Cloud и используйте healthcheck.py для диагностики"
echo ""
echo "Примечание: Не забудьте указать следующие настройки в Streamlit Cloud:"
echo "   - Main file path: app_cloud.py"
echo "   - Python version: 3.11"
echo "   - Requirements file: requirements.txt (копия requirements-clean.txt)"
echo ""
echo "Для возврата к локальной разработке, восстановите оригинальный config.toml:"
echo "./restore_local_config.sh"