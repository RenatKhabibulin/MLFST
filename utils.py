import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import uuid
import os
from pathlib import Path
import base64

def load_css():
    """Load custom CSS to enhance the default Streamlit theme"""
    # Add custom CSS for sidebar buttons styling
    st.markdown("""
    <style>
    /* Sidebar button styling */
    .stButton > button {
        background-color: rgba(70, 70, 70, 0.7);
        color: white;
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px 15px;
        font-weight: 500;
        box-shadow: 0px 3px 5px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    /* Hover effect */
    .stButton > button:hover {
        background-color: rgba(80, 80, 80, 0.9);
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0px 5px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }

    /* Active effect */
    .stButton > button:active {
        background-color: rgba(60, 60, 60, 0.9);
        box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.2);
        transform: translateY(0px);
    }
    
    /* Progress indicators in sidebar buttons */
    .sidebar-progress {
        color: #4CAF50;
        font-weight: bold;
    }
    
    /* Improved styling for math formulas in expanders */
    .streamlit-expanderContent {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
    }
    
    /* Make math formulas more readable */
    .katex {
        font-size: 1.1em !important;
    }
    
    /* Space between math blocks */
    .katex-display {
        margin: 1.5em 0 !important;
    }
    
    /* Styling for the math descriptions sections */
    .streamlit-expanderContent h3 {
        color: #FF5722;
        margin-top: 1em;
        margin-bottom: 0.5em;
        font-weight: 600;
    }
    
    .streamlit-expanderContent h4 {
        color: #1976D2;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
        font-weight: 500;
    }
    
    /* Better styling for plots and charts */
    .js-plotly-plot {
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border-radius: 8px !important;
        margin: 1em 0 !important;
    }
    
    /* Modern styling for all plots and charts */
    .js-plotly-plot .plotly .main-svg {
        border-radius: 8px !important;
    }
    
    /* Enhanced colors for Plotly charts */
    .js-plotly-plot .plotly .scatter .lines {
        stroke-width: 3px !important;
    }
    
    /* Make matplotlib figures more vibrant */
    .stImage img {
        border-radius: 8px !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    /* General styling for data-related elements */
    .element-container iframe {
        border-radius: 8px !important;
    }
    
    /* Styling for plotly legends */
    .js-plotly-plot .legend .bg {
        opacity: 0.8 !important;
    }
    
    /* Make charts responsive */
    @media (max-width: 768px) {
        .js-plotly-plot {
            height: auto !important;
        }
    }
    
    /* Module buttons specific styling */
    [data-testid="stSidebar"] .stButton > button {
        text-align: left;
        width: 100%;
        white-space: normal;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] .stMarkdown h1, 
    [data-testid="stSidebar"] .stMarkdown h2, 
    [data-testid="stSidebar"] .stMarkdown h3 {
        margin-top: 20px;
        padding-bottom: 5px;
        border-bottom: 1px solid rgba(250, 250, 250, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def load_dataset(filename):
    """
    Load a dataset from the data directory
    Returns a pandas DataFrame
    """
    data_path = Path("data") / filename
    
    if data_path.exists():
        return pd.read_csv(data_path)
    else:
        st.error(f"Dataset file {filename} not found")
        return pd.DataFrame()

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix using matplotlib
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a table to display the confusion matrix
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
    ax.set_title('Confusion Matrix')
    
    # Add axis labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    plt.tight_layout()
    
    return fig

def check_password_hash(hashed_password, user_password):
    """Check a stored password against one provided by user"""
    if ':' not in hashed_password:
        return False
    
    password, salt = hashed_password.split(':')
    return password == hashlib.sha256(salt.encode() + user_password.encode()).hexdigest()

def create_module_quiz(questions, module_id):
    """
    Create a quiz with multiple choice questions
    Returns the score as a percentage
    
    Parameters:
    - questions: list of dict objects with 'question', 'options', and 'correct' keys
    - module_id: the ID of the module for tracking progress
    """
    if not questions:
        st.warning("No questions available for this quiz")
        return 0
    
    # Add test header with enhanced styling
    st.markdown("""
    <style>
    .module-test-header {
        background: linear-gradient(to right, #FF5722, #FF9800);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 30px 0 20px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .test-introduction {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 15px 0;
    }
    .question-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
        border-left: 4px solid #FF9800;
    }
    .question-number {
        display: inline-block;
        background-color: #FF9800;
        color: white;
        width: 32px;
        height: 32px;
        line-height: 32px;
        text-align: center;
        border-radius: 50%;
        margin-right: 10px;
        font-weight: bold;
    }
    .question-content {
        font-size: 18px;
        color: #333;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="module-test-header">', unsafe_allow_html=True)
    st.markdown('# 📝 Module Assessment', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="test-introduction">', unsafe_allow_html=True)
    st.markdown("""
    This assessment will test your understanding of the module concepts. Select the best answer for each question.
    
    - The quiz contains **{}** questions
    - You need to score at least **70%** to pass
    - Your results and progress will be saved automatically
    """.format(len(questions)), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state for quiz if not exists
    if f'quiz_answers_{module_id}' not in st.session_state:
        st.session_state[f'quiz_answers_{module_id}'] = [-1] * len(questions)
    
    if f'quiz_submitted_{module_id}' not in st.session_state:
        st.session_state[f'quiz_submitted_{module_id}'] = False
    
    # Display questions
    for i, q in enumerate(questions):
        # Create card for each question
        st.markdown(f'<div class="question-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="question-content"><span class="question-number">{i+1}</span>{q["question"]}</p>', unsafe_allow_html=True)
        
        # Radio buttons for options
        answer = st.radio(
            f"Select your answer:",
            options=q['options'],
            index=st.session_state[f'quiz_answers_{module_id}'][i] 
                if st.session_state[f'quiz_answers_{module_id}'][i] >= 0 
                else 0,
            key=f"q_{module_id}_{i}"
        )
        
        # Save answer
        st.session_state[f'quiz_answers_{module_id}'][i] = q['options'].index(answer)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a clean separation before buttons
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Create better button layout
    st.markdown("""
    <style>
    .quiz-button-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        margin-bottom: 20px;
        align-items: center;
        max-width: 450px;
        margin-left: auto;
        margin-right: auto;
    }
    .quiz-button-container .stButton button {
        width: 100%;
        padding: 12px 20px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15) !important;
    }
    .submit-button button {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
    }
    .reset-button button {
        background-color: #f44336 !important;
        color: white !important;
        border: none !important;
    }
    .submit-button button:hover {
        background-color: #45a049 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transform: translateY(-2px) !important;
    }
    .reset-button button:hover {
        background-color: #d32f2f !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Button container - вертикальное расположение кнопок одна под другой
    st.markdown('<div class="quiz-button-container">', unsafe_allow_html=True)
    
    # Перемещаем кнопку Submit Quiz наверх
    st.markdown('<div class="submit-button" style="margin-bottom: 15px;">', unsafe_allow_html=True)
    if st.button("📝 Submit Quiz", key=f"submit_quiz_{module_id}", use_container_width=True):
        st.session_state[f'quiz_submitted_{module_id}'] = True
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Reset Quiz кнопка под Submit Quiz
    st.markdown('<div class="reset-button">', unsafe_allow_html=True)
    if st.button("🔄 Reset Quiz", key=f"reset_quiz_{module_id}", use_container_width=True):
        st.session_state[f'quiz_answers_{module_id}'] = [-1] * len(questions)
        st.session_state[f'quiz_submitted_{module_id}'] = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Calculate and display results if submitted
    if st.session_state[f'quiz_submitted_{module_id}']:
        correct_count = 0
        
        # Add results styling
        st.markdown("""
        <style>
        .results-header {
            background: linear-gradient(to right, #2196F3, #03A9F4);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin: 30px 0 20px 0;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .results-summary {
            display: flex;
            justify-content: space-around;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .result-card {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            flex: 1;
            margin: 0 10px;
        }
        .score-value {
            font-size: 42px;
            font-weight: bold;
            margin: 10px 0;
        }
        .correct-count {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .score-label, .count-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .pass-fail {
            font-size: 28px;
            font-weight: bold;
            margin: 15px 0;
            padding: 10px;
            border-radius: 10px;
        }
        .pass {
            background-color: #E8F5E9;
            color: #2E7D32;
        }
        .fail {
            background-color: #FFEBEE;
            color: #C62828;
        }
        .answer-review {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .answer-correct {
            background-color: #E8F5E9;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }
        .answer-incorrect {
            background-color: #FFEBEE;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid #F44336;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Calculate results
        for i, q in enumerate(questions):
            user_answer_idx = st.session_state[f'quiz_answers_{module_id}'][i]
            correct_idx = q['options'].index(q['correct'])
            
            if user_answer_idx == correct_idx:
                correct_count += 1
        
        score = (correct_count / len(questions)) * 100
        
        # Display results header
        st.markdown('<div class="results-header">', unsafe_allow_html=True)
        st.markdown('# 📊 Assessment Results', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results summary cards
        st.markdown('<div class="results-summary">', unsafe_allow_html=True)
        
        # Score card
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="score-label">Your Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-value">{score:.1f}%</div>', unsafe_allow_html=True)
        
        # Show progress bar inside card
        st.progress(score / 100)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Correct answers card
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="count-label">Correct Answers</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="correct-count">{correct_count} / {len(questions)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Pass/Fail card
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if score >= 70:
            st.markdown('<div class="pass-fail pass">PASSED</div>', unsafe_allow_html=True)
            st.markdown('🎉 Great job! You\'ve passed the assessment.', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pass-fail fail">NEEDS IMPROVEMENT</div>', unsafe_allow_html=True)
            st.markdown('📚 Keep studying! You need 70% to pass.', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed answer review
        st.markdown('<div class="answer-review">', unsafe_allow_html=True)
        st.markdown('## Detailed Answer Review', unsafe_allow_html=True)
        
        for i, q in enumerate(questions):
            user_answer_idx = st.session_state[f'quiz_answers_{module_id}'][i]
            correct_idx = q['options'].index(q['correct'])
            
            if user_answer_idx == correct_idx:
                st.markdown(f'<div class="answer-correct">', unsafe_allow_html=True)
                st.markdown(f'<p><span class="question-number">{i+1}</span> <strong>Correct!</strong> ✓</p>', unsafe_allow_html=True)
                st.markdown(f'<p><strong>Question:</strong> {q["question"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p><strong>Your answer:</strong> {q["options"][user_answer_idx]}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="answer-incorrect">', unsafe_allow_html=True)
                st.markdown(f'<p><span class="question-number">{i+1}</span> <strong>Incorrect</strong> ✗</p>', unsafe_allow_html=True)
                st.markdown(f'<p><strong>Question:</strong> {q["question"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p><strong>Your answer:</strong> {q["options"][user_answer_idx]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p><strong>Correct answer:</strong> {q["correct"]}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Return score for tracking
        return score / 100
    
    return 0

def create_code_editor(code_sample, language):
    """Create a code editor with the provided sample code"""
    edited_code = st.text_area(
        "Edit the code below:",
        value=code_sample,
        height=300
    )
    
    if st.button("Run Code", key=f"run_code_{language}_{hash(code_sample)}"):
        st.write("Code output:")
        
        # Display a message that in this demo environment, 
        # code execution is simulated
        with st.container():
            st.info("Code execution is simulated in this demo environment.")
            
            # For Python, we can show what the output would look like
            if language.lower() == "python":
                st.code("# Simulated output would appear here", language="bash")
            else:
                st.code("/* Simulated output would appear here */", language="bash")
    
    return edited_code
