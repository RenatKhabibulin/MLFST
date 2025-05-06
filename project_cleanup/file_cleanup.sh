#!/bin/bash

# Скрипт для очистки проекта перед загрузкой на GitHub

# Удаление временных файлов и резервных копий
echo "Удаление временных файлов и резервных копий..."
find . -name "*.bak" -type f -delete
find . -name "*.backup" -type f -delete
find . -name "*.bk*" -type f -delete
find . -name "*~" -type f -delete

# Удаление кэша Python
echo "Удаление кэша Python..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete

# Удаление системных и конфигурационных директорий
echo "Удаление системных и конфигурационных директорий..."
rm -rf .cache/ .pythonlibs/ .local/ .config/ .upm/

# Удаление специфичных для Replit файлов
echo "Удаление специфичных для Replit файлов..."
rm -f .replit .replit.nix .breakpoints

# Удаление баз данных (если требуется сохранить данные закомментируйте эти строки)
# echo "Удаление файлов баз данных..."
# find . -name "*.db" -type f -delete
# find . -name "*.sqlite" -type f -delete
# find . -name "*.sqlite3" -type f -delete

# Очистка директории .git от крупных файлов
echo "Очистка директории .git..."
git gc

echo "Очистка завершена!"
echo "Проверьте структуру проекта перед загрузкой на GitHub."