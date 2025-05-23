# Руководство по деплою "ML FireSafety Tutor"

## Содержание

1. [Требования для деплоя](#требования-для-деплоя)
2. [Подготовка к деплою](#подготовка-к-деплою)
3. [Деплой на Streamlit Cloud](#деплой-на-streamlit-cloud)
4. [Решение проблем](#решение-проблем)
5. [Возврат к локальной разработке](#возврат-к-локальной-разработке)

## Требования для деплоя

Для успешного деплоя приложения "ML FireSafety Tutor" на Streamlit Cloud вам потребуется:

1. **Аккаунт на Streamlit Cloud** (https://streamlit.io/cloud)
2. **GitHub-репозиторий** с кодом приложения
3. **Python 3.11** (не использовать Python 3.12)
4. **Правильные настройки конфигурации** для портов и зависимостей

## Подготовка к деплою

Проект содержит скрипт для автоматической подготовки к деплою. Выполните:

```bash
./prepare_for_deploy.sh
```

Скрипт выполнит следующие действия:
1. Скопирует `.streamlit/config_cloud.toml` в `.streamlit/config.toml` (настройка порта 8501)
2. Проверит наличие `requirements-clean.txt` и `app_cloud.py`
3. Напомнит о необходимости выбора Python 3.11

Если вы хотите выполнить эти действия вручную:

```bash
# Сохраняем текущую конфигурацию (если ещё не сделано)
cp .streamlit/config.toml .streamlit/config.toml.backup

# Копируем конфигурацию для Streamlit Cloud
cp .streamlit/config_cloud.toml .streamlit/config.toml
```

### Использование скрипта запуска entry.sh

Для улучшения совместимости с Streamlit Cloud создан специальный скрипт `entry.sh`, который обеспечивает запуск приложения на правильном порту. 

Файл `Procfile` в корне проекта указывает Streamlit Cloud использовать этот скрипт:
```
web: bash entry.sh
```

Это гарантирует, что приложение будет запущено на порту 8501, даже если конфигурация в `.streamlit/config.toml` указывает другой порт.

## Деплой на Streamlit Cloud

1. **Залейте подготовленный код в GitHub-репозиторий**
   - Убедитесь, что все изменения после выполнения скрипта подготовки закоммичены

2. **Создайте новое приложение в Streamlit Cloud**
   - Войдите в аккаунт Streamlit Cloud
   - Нажмите "New app"
   - Выберите ваш репозиторий

3. **Настройте параметры деплоя:**
   - Main file path: `app_cloud.py`
   - Python version: `3.11`
   - Requirements file: `requirements-clean.txt`

4. **Нажмите "Deploy!"**

5. **Дождитесь завершения сборки и деплоя**
   - Процесс может занять несколько минут
   - Проверьте логи на наличие ошибок

## Решение проблем

### Ошибка "No module named 'distutils'"
- Используйте Python 3.11 вместо 3.12
- Убедитесь, что в `requirements-clean.txt` указаны совместимые версии пакетов

### Ошибка "Connection refused on port 8501"
- Проверьте, что используется конфигурация из `config_cloud.toml`
- Убедитесь, что приложение не привязано жестко к порту 5000 в коде

### Другие ошибки
- Просмотрите полные логи деплоя в Streamlit Cloud
- Проверьте документацию в папке `project_cleanup/` для дополнительных рекомендаций

## Возврат к локальной разработке

После деплоя вы можете вернуться к локальной конфигурации:

```bash
./restore_local_config.sh
```

Этот скрипт восстановит конфигурацию для порта 5000, необходимую для локальной разработки.

## Дополнительная документация

Дополнительная документация по деплою доступна в папке `project_cleanup/`:

- `deployment_checklist.md` - подробный чек-лист для деплоя
- `python312_compatibility.md` - информация о совместимости с Python 3.12
- `port_configuration.md` - подробности о настройке портов
- `local_vs_cloud.md` - различия между локальным запуском и Streamlit Cloud
- `streamlit_cloud_deployment.md` - дополнительные рекомендации по деплою