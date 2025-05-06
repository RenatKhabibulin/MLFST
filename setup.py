from setuptools import setup, find_packages

setup(
    name="mlfiresafetytutor",
    version="0.1.0",
    description="ML FireSafetyTutor - Interactive platform for learning ML in fire safety applications",
    author="Author",
    packages=find_packages(include=["mlfiresafetytutor", "mlfiresafetytutor.*", "modules", "modules.*"]),
    py_modules=["app", "auth", "database", "ml_utils", "utils", "visualization", "evacuation_example", "streamlit_app"],
    install_requires=[
        "streamlit>=1.45.0",
        "pandas>=2.2.3",
        "numpy>=2.2.5",
        "matplotlib>=3.10.1",
        "plotly>=6.0.1",
        "scikit-learn>=1.6.1",
        "sqlalchemy>=2.0.40",
        "psycopg2-binary>=2.9.10",
        "scipy>=1.15.2",
        "seaborn>=0.13.2",
        "anthropic>=0.50.0",
        "openai>=1.77.0",
    ],
)