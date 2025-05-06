# ML FireSafety Tutor

An interactive Streamlit application providing educational content on machine learning applications in fire safety.

## Features

- Interactive learning modules covering machine learning concepts
- Practical examples of ML applications in fire safety
- Hands-on exercises with realistic data
- Progress tracking and assessment
- User authentication system
- Visualization of fire safety scenarios

## Installation

1. Clone this repository
```bash
git clone https://github.com/yourusername/mlfiresafetytutor.git
cd mlfiresafetytutor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run app.py
```

## Module Structure

The application contains the following learning modules:

1. Introduction to ML in Fire Safety
2. Data Preparation and Processing
3. Supervised Learning Algorithms
4. Unsupervised Learning Algorithms
5. Model Evaluation
6. Deep Learning for Fire Detection
7. AI Agents in Fire Safety

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- plotly
- scikit-learn
- psycopg2-binary
- sqlalchemy
- scipy
- seaborn
- anthropic
- openai

## Database Setup

The application uses PostgreSQL for storing user data and progress. You'll need to set up a PostgreSQL database and configure the connection details in `.env` file (create one based on `.env.example`).

## Documentation

For more detailed documentation, please refer to `user_guide.html` included in the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## Deployment

For deploying on Streamlit Cloud, use the following settings:
- Main file path: `app_cloud.py`
- Python version: 3.11
- Requirements file: `requirements.txt`

For full functionality, it's recommended to deploy this application on your own server infrastructure.