# Инструкция по деплою на Streamlit Cloud

## Проблема: "No module named 'distutils'"

Эта ошибка часто возникает в следующих случаях:
1. При использовании сложной пакетной структуры проекта с `pyproject.toml`
2. При использовании Python 3.12, в котором модуль distutils был удалён
3. При использовании старых версий пакетов, зависящих от distutils (например, numpy==1.24.4)

## Решение 1: Упрощенный подход без пакетной структуры

1. Удалите или переименуйте файл `pyproject.toml` (например, в `pyproject.toml.old`)
2. Используйте упрощенный файл `app_cloud.py` как точку входа
3. В настройках Streamlit Cloud укажите:
   - Main file path: `app_cloud.py`
   - Python version: 3.10 или 3.11 (не используйте Python 3.12!)
   - Requirements file: `requirements-clean.txt` (с обновленными пакетами)

## Решение 2: Создание отдельного репозитория для деплоя

1. Создайте новый репозиторий для деплоя, содержащий только необходимые файлы:
   - `app_cloud.py` (переименованный в `app.py`)
   - `requirements.txt` (очищенная версия)
   - Необходимые ресурсы и модули

2. В настройках Streamlit Cloud укажите:
   - Main file path: `app.py`
   - Python version: 3.11
   - Requirements file: `requirements.txt`

## Решение 3: Использование GitHub Pages вместо Streamlit Cloud

Если деплой на Streamlit Cloud продолжает вызывать проблемы, рассмотрите возможность использовать GitHub Pages для статической версии вашего приложения:

1. Создайте директорию `docs/` в корне репозитория
2. Добавьте в неё HTML-версию вашего приложения (можно экспортировать из Streamlit с помощью снимков экрана или используя библиотеки для конвертации)
3. Включите GitHub Pages в настройках репозитория, указав директорию `docs/` как источник

## Ключевые моменты для успешного деплоя

1. **Тщательно очищенный файл requirements.txt** - только необходимые зависимости без дубликатов
2. **Упрощенная структура файлов** - без сложной пакетной структуры
3. **Правильная конфигурация точки входа** - один чёткий "главный" файл
4. **Совместимые версии Python и библиотек** - тщательно проверьте совместимость версий
5. **Правильная настройка доступа к данным** - используйте относительные пути