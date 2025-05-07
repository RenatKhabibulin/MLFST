# Оптимизация производительности Streamlit приложения

В этом документе приведены рекомендации по оптимизации производительности и решению проблем с памятью в приложении ML FireSafety Tutor.

## Проблемы с памятью и ресурсами

Streamlit-приложения могут столкнуться с проблемами потребления памяти и ресурсов, особенно при работе с большими наборами данных или сложными моделями машинного обучения. Основные симптомы:

- Медленная загрузка страниц
- Ошибки Out of Memory (OOM)
- Сбои приложения при повышенной нагрузке
- Высокое использование CPU/RAM
- Таймауты при длительных операциях

## Стратегии оптимизации

### 1. Кэширование с помощью декораторов Streamlit

Streamlit предлагает два мощных декоратора для кэширования:

#### `@st.cache_data`

Кэширует возвращаемые значения функции, идеально подходит для:
- Загрузки и предобработки данных
- Выполнения дорогостоящих вычислений
- Преобразования данных

```python
@st.cache_data
def load_dataset(filename):
    """
    Загружает и кэширует датасет, чтобы избежать повторной загрузки
    """
    return pd.read_csv(f"data/{filename}")

# Пример использования
df = load_dataset("fire_incidents.csv")
```

#### `@st.cache_resource`

Кэширует ресурсы, которые не должны быть повторно созданы, идеально для:
- Моделей машинного обучения
- Подключений к базам данных
- Сложных объектов

```python
@st.cache_resource
def get_trained_model():
    """
    Загружает или тренирует модель машинного обучения один раз
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Пример использования
model = get_trained_model()
predictions = model.predict(X_test)
```

### 2. Оптимизация работы с данными

- **Уменьшите размер обрабатываемых данных**: Используйте выборки данных, когда возможно
- **Агрегируйте данные предварительно**: Вместо хранения каждой точки данных, агрегируйте их
- **Используйте эффективные типы данных**: Например, категориальные типы в pandas
- **Фильтруйте данные раньше**: Загружайте только необходимые столбцы и строки

```python
# Вместо:
df = pd.read_csv("большой_файл.csv")
filtered_df = df[df['column'] > threshold]

# Используйте:
df = pd.read_csv("большой_файл.csv", usecols=['column', 'нужная_колонка'])
filtered_df = df[df['column'] > threshold]
```

### 3. Применение к файлам ML FireSafety Tutor

#### utils.py

```python
# Было:
def load_dataset(filename):
    """
    Load a dataset from the data directory
    Returns a pandas DataFrame
    """
    return pd.read_csv(f"data/{filename}")

# Стало:
@st.cache_data
def load_dataset(filename):
    """
    Load a dataset from the data directory with caching
    Returns a pandas DataFrame
    """
    return pd.read_csv(f"data/{filename}")
```

#### ml_utils.py

```python
# Было:
def train_random_forest(X, y):
    """
    Train a random forest classifier and return predictions and metrics
    """
    model = RandomForestClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'report': classification_report(y, y_pred)
    }
    return model, y_pred, metrics

# Стало:
@st.cache_resource
def train_random_forest(X, y, _hash=None):
    """
    Train a random forest classifier and return predictions and metrics
    Uses caching to avoid retraining the model unnecessarily
    
    Parameters:
    - X: features array
    - y: target array
    - _hash: optional parameter for cache invalidation
    """
    model = RandomForestClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'report': classification_report(y, y_pred)
    }
    return model, y_pred, metrics
```

#### visualization.py

```python
# Было:
def plot_clustering_example(df):
    """
    Demonstrate K-means clustering on sensor data
    """
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_numeric = df.select_dtypes(include=[np.number])
    columns = df_numeric.columns[:2]  # Take first two numeric columns
    X = df_numeric[columns].values
    kmeans.fit(X)
    labels = kmeans.labels_
    # Plotting code...

# Стало:
@st.cache_data
def perform_clustering(df):
    """
    Perform clustering and return results
    """
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_numeric = df.select_dtypes(include=[np.number])
    columns = df_numeric.columns[:2]  # Take first two numeric columns
    X = df_numeric[columns].values
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_, X, columns

def plot_clustering_example(df):
    """
    Demonstrate K-means clustering on sensor data with caching
    """
    labels, centers, X, columns = perform_clustering(df)
    # Plotting code using these cached values...
```

### 4. Ограничение размера данных

- **Используйте выборки данных для демонстрации**: Для обучающих примеров достаточно 1000-10000 строк
- **Создавайте агрегированные наборы данных**: Предварительно агрегируйте данные для визуализаций
- **Фильтруйте на этапе загрузки**: Используйте аргументы `nrows` и `usecols` в `pd.read_csv`

```python
@st.cache_data
def load_limited_dataset(filename, max_rows=10000):
    """
    Load a limited dataset from the data directory
    """
    df = pd.read_csv(f"data/{filename}", nrows=max_rows)
    return df
```

### 5. Ленивая загрузка модулей

Загружайте тяжелые компоненты только когда они запрашиваются пользователем:

```python
# Вместо:
import heavy_module_1
import heavy_module_2
import heavy_module_3

# Используйте:
if selected_page == "Module 1":
    import heavy_module_1
    heavy_module_1.render()
```

### 6. Мониторинг использования памяти

Добавьте функцию мониторинга памяти для отладки:

```python
import psutil
import os

def show_memory_usage():
    """Показывает текущее использование памяти процессом Python"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_usage_mb = mem_info.rss / 1024 / 1024
    st.sidebar.text(f"Memory usage: {memory_usage_mb:.2f} MB")
```

## Рекомендации для ML FireSafety Tutor

1. **Добавьте кэширование во все функции загрузки данных** используя `@st.cache_data`
2. **Кэшируйте модели ML** с помощью `@st.cache_resource`
3. **Оптимизируйте визуализацию**, разделив вычисления и отображение
4. **Ограничьте размеры всех используемых наборов данных**
5. **Добавьте опцию для отображения использования памяти** в режиме разработки
6. **Ленивую загрузку модулей** применяйте для тяжеловесных компонентов

## Дополнительные оптимизации для Streamlit Cloud

При деплое в Streamlit Cloud дополнительно рекомендуется:

1. **Уменьшите размер зависимостей** в `requirements-clean.txt`
2. **Оптимизируйте загрузку больших файлов** с использованием потоковой загрузки
3. **Установите таймауты для длительных операций**
4. **Используйте сжатие изображений** перед отображением

Более подробные рекомендации по оптимизации производительности можно найти в [официальной документации Streamlit](https://docs.streamlit.io/library/advanced-features/caching).