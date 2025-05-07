# Настройка портов для Streamlit Cloud

## Проблема с портом

При деплое на Streamlit Cloud обычно используется порт 8501, однако в вашей конфигурации приложение запускается на порту 5000. Это приводит к ошибке:
```
The service has encountered an error while checking the health of the Streamlit app: 
Get "http://localhost:8501/healthz": dial tcp 127.0.0.1:8501: connect: connection refused
```

## Решение 1: Настройка конфигурации Streamlit

Создайте или отредактируйте файл `.streamlit/config.toml` для использования правильного порта при деплое:

```toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
```

## Решение 2: Использование порта 8501 в app_cloud.py

Для app_cloud.py можно добавить код для явного указания порта 8501:

```python
import streamlit as st
import os

# Настройка порта
port = int(os.environ.get("PORT", 8501))
```

## Решение 3: Использование переменных окружения

В Streamlit Cloud доступны переменные окружения, которые можно использовать для определения порта:

```python
import os
import streamlit as st

# Получение порта из переменной окружения или использование порта по умолчанию
port = int(os.environ.get("PORT", 8501))

# Ваш код приложения
st.title("ML FireSafety Tutor")
# ...
```

## Рекомендации для локального и облачного запуска

Лучший подход - использовать настройки, которые работают как локально, так и в облаке:

1. В `.streamlit/config.toml`:
```toml
[server]
headless = true
enableCORS = false
```

2. В коде приложения:
```python
import os
import streamlit as st

# Порт может быть передан через переменную окружения
port = int(os.environ.get("PORT", 8501))

# Остальной код приложения
# ...
```

Это позволит приложению адаптироваться к разным окружениям без необходимости изменять код.