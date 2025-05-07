# Локальный запуск vs Streamlit Cloud

## Разница в конфигурации портов

При локальной разработке и при деплое на Streamlit Cloud возникают различия в конфигурации портов, что может вызывать проблемы с запуском приложения. Этот документ объясняет разницу и предлагает решения.

### Локальный запуск (в среде разработки)

При локальном запуске в Replit мы используем порт **5000**, так как это порт, который разрешен для доступа извне:

```bash
streamlit run app.py --server.port 5000
```

Соответствующая конфигурация в `.streamlit/config.toml`:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
```

### Запуск в Streamlit Cloud

Streamlit Cloud ожидает, что приложение будет запущено на порту **8501** (стандартный порт Streamlit):

```bash
streamlit run app.py
# или явно указать порт
streamlit run app.py --server.port 8501
```

Соответствующая конфигурация в `.streamlit/config_cloud.toml`:
```toml
[server]
headless = true
port = 8501
enableCORS = false
```

## Решение проблемы

Чтобы обеспечить работу приложения как локально, так и в облаке, реализовано несколько подходов:

1. **Создание отдельных конфигурационных файлов:**
   - `.streamlit/config.toml` - для локального запуска (порт 5000)
   - `.streamlit/config_cloud.toml` - для Streamlit Cloud (порт 8501)

2. **Динамическое определение порта в коде:**
   ```python
   import os
   PORT = int(os.environ.get("PORT", 8501))
   ```

3. **Подготовка специального файла для деплоя:**
   - `app_cloud.py` - упрощенная версия приложения для Streamlit Cloud

## Инструкция по деплою

1. Скопируйте `.streamlit/config_cloud.toml` в `.streamlit/config.toml`
2. В интерфейсе Streamlit Cloud укажите:
   - Main file path: `app_cloud.py`
   - Python version: `3.11`
   - Requirements file: `requirements-clean.txt`

## Важные замечания

- Не используйте Python 3.12 для деплоя, так как в нем отсутствует модуль `distutils`, необходимый для установки некоторых пакетов.
- Проверьте, что в файле `app_cloud.py` нет жестко заданного порта 5000.
- Если возникают проблемы с запуском, проверьте логи Streamlit Cloud для выявления конкретных ошибок.