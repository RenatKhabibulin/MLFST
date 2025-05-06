# Структура проекта ML FireSafetyTutor

Эта директория содержит информацию о структуре проекта и рекомендации по очистке для деплоя на GitHub.

## Основные файлы приложения (должны быть сохранены):

- `app.py` - Основной файл приложения
- `app_cloud.py` - Упрощенная версия для деплоя на Streamlit Cloud
- `streamlit_app.py` - Входная точка для Streamlit Cloud
- `auth.py` - Функции аутентификации
- `database.py` - Работа с базой данных
- `ml_utils.py` - Утилиты для машинного обучения
- `utils.py` - Вспомогательные функции
- `visualization.py` - Функции визуализации
- `evacuation_example.py` - Пример эвакуации

## Модули (должны быть сохранены):

- `modules/` - Директория с учебными модулями

## Файлы настроек и документации:

- `README.md` - Основная документация
- `LICENSE` - Лицензия MIT
- `CONTRIBUTING.md` - Правила участия в разработке
- `requirements.txt` - Зависимости (используйте очищенную версию)
- `.streamlit/config.toml` - Конфигурация Streamlit
- `.gitignore` - Настройки git

## Файлы, которые можно удалить:

- `*.bak`, `*.backup`, `*.bk*` - Резервные копии
- `pyproject.toml.backup*` - Резервные копии файла настроек
- `.cache/`, `.pythonlibs/`, `.local/`, `.config/`, `.upm/` - Кэш и локальные библиотеки
- `.replit`, `.replit.nix` - Специфичные файлы для Replit
- `__pycache__/` - Кэш Python
- `readme.txt` - Дублирует README.md
- `uv.lock` - Временный файл

## Рекомендации по деплою на Streamlit Cloud:

1. Используйте упрощенную версию `app_cloud.py` как точку входа
2. В настройках деплоя укажите:
   - Main file path: `app_cloud.py`
   - Python version: 3.11
   - Requirements file: `requirements.txt`

## Файлы конфигурации:

В случае проблем с деплоем из-за `pyproject.toml`, можно использовать более простой подход без пакетной структуры, просто указав `app_cloud.py` как точку входа.