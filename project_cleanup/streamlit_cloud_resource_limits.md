# Проблема ресурсов в Streamlit Cloud и решение

## Проблема

При деплое полной версии приложения ML FireSafety Tutor в Streamlit Cloud возникает ошибка:

```
The service has encountered an error while checking the health of the Streamlit app:
Get "http://localhost:8501/healthz": dial tcp 127.0.0.1:8501: connect: connection refused
```

После тщательного анализа выяснилась истинная причина проблемы: **ограничение ресурсов в бесплатном тире Streamlit Cloud**.

## Почему приложение работает на Replit, но не на Streamlit Cloud?

1. **Разные ограничения ресурсов:**
   - Replit предоставляет более щедрые ресурсы (память и CPU)
   - Бесплатный тир Streamlit Cloud имеет жесткие ограничения (~1GB RAM)

2. **Разный процесс развертывания:**
   - Replit: более гибкая система контейнеризации с управлением зависимостями
   - Streamlit Cloud: оптимизирован для легких приложений с жесткими проверками состояния

3. **Использование ресурсоемких библиотек:**
   Наше приложение использует:
   - Scikit-learn, NumPy, Pandas (тяжелые библиотеки обработки данных)
   - Matplotlib, Plotly (визуализация данных)
   - SQLAlchemy, PostgreSQL (работа с базой данных)
   - Streamlit (интерфейс пользователя)

## Решение: УЛЬТРАЛЕГКАЯ версия

Для развертывания в Streamlit Cloud создана специальная ультралегкая демо-версия приложения:

1. **app_cloud_ultralight.py**:
   - Импортирует только Streamlit (без тяжелых ML библиотек)
   - Показывает интерфейс приложения без функциональности ML
   - Использует статические изображения вместо динамических графиков

2. **requirements-ultralight.txt**:
   - Содержит только streamlit (без pandas, sklearn и т.д.)
   - Минимизирует использование памяти

3. **entry.sh**:
   - Запускает ультралегкую версию
   - Добавляет диагностику системных ресурсов

4. **prepare_for_deploy.sh**:
   - Подготавливает проект к деплою ультралегкой версии
   - Выполняет необходимые проверки

## Когда использовать полную или легкую версию?

1. **Полная версия (app.py)**:
   - Локальная разработка и тестирование
   - Развертывание на серверах с достаточными ресурсами
   - Полная функциональность, включая ML и визуализацию данных

2. **Ультралегкая версия (app_cloud_ultralight.py)**:
   - Деплой в бесплатном тире Streamlit Cloud
   - Демонстрационные цели
   - Облегченная визуализация и интерфейс без ML функциональности

## Другие опции для полной версии

Если вам необходима полная функциональность в облаке, рассмотрите:

1. **Платный тир Streamlit Cloud**:
   - Больше ресурсов для запуска полной версии
   - Поддержка всех функций

2. **Другие платформы**:
   - Heroku (с платным тиром)
   - AWS, GCP, Azure (при наличии бюджета)
   - PythonAnywhere (подходит для образовательных целей)

3. **Микросервисная архитектура**:
   - Разделение на легкий интерфейс (Streamlit Cloud)
   - Тяжелое ML API (другой хостинг)

## Как использовать ультралегкую версию

```bash
./prepare_for_deploy.sh
# Затем загрузите на Streamlit Cloud
```

После деплоя в Streamlit Cloud нужно:
- Указать app_cloud_ultralight.py как точку входа
- Выбрать Python 3.11
- Использовать requirements.txt (минимальные зависимости)

## Возврат к полной версии для локальной разработки

```bash
./restore_local_config.sh
```