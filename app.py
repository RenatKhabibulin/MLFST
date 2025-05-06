import streamlit as st
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Import visualization tools and styles
from visualization import (
    apply_firesafety_theme_to_plotly, 
    apply_firesafety_theme_to_matplotlib,
    FIRESAFETY_COLORS,
    FIRESAFETY_SEQUENTIAL,
    FIRESAFETY_CATEGORICAL
)

# Import custom modules
from auth import authenticate_user, show_login_page, show_registration_page, show_reset_password_page
from database import init_db, get_user_progress, update_user_progress, get_user_test_results
import modules.intro
import modules.data_prep
import modules.supervised
import modules.unsupervised
import modules.deep_learning
import modules.model_eval
import modules.ai_agents
# Temporarily comment out missing modules
# import modules.model_interpret
# import modules.deployment
from utils import check_password_hash, load_css

# Set page config
st.set_page_config(
    page_title="ML FireSafe - Learn ML for Fire Safety",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database
init_db()

# Initialize session state variables
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'current_module' not in st.session_state:
    st.session_state.current_module = 0
if 'module_progress' not in st.session_state:
    st.session_state.module_progress = {}
if 'is_guest' not in st.session_state:
    st.session_state.is_guest = False

# Define modules
modules = [
    {'name': 'Introduction to ML in Fire Safety', 'function': modules.intro},
    {'name': 'Data Preparation and Processing', 'function': modules.data_prep},
    {'name': 'Supervised Learning Algorithms', 'function': modules.supervised},
    {'name': 'Unsupervised Learning Algorithms', 'function': modules.unsupervised},
    {'name': 'Deep Learning', 'function': modules.deep_learning},
    {'name': 'Model Evaluation', 'function': modules.model_eval},
    {'name': 'AI Agents in Fire Safety', 'function': modules.ai_agents},
    # Temporarily comment out missing modules
    # {'name': 'Model Interpretation', 'function': modules.model_interpret},
    # {'name': 'Deployment in Fire Safety', 'function': modules.deployment},
]

def show_main_app():
    st.sidebar.title("ML FireSafetyTutor")
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#f0f2f6, #e6e9ef);
        color: black;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Regular buttons */
    .stButton>button {
        background-color: #ff5722;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
    }
    /* Semi-transparent sidebar buttons */
    .sidebar .stButton>button {
        background-color: rgba(255, 87, 34, 0.7);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    /* Hover effect for sidebar buttons */
    .sidebar .stButton>button:hover {
        background-color: rgba(255, 87, 34, 0.9);
    }
    .stButton>button {
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #e64a19;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: #333;
        font-weight: bold;
    }
    h1 {
        border-bottom: 2px solid #ff5722;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .stProgress > div > div {
        background-color: #ff5722;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # User info and logout
    st.sidebar.write(f"Welcome, {st.session_state.user_name}")
    
    # User guide button
    guide_container = st.sidebar.container()
    if st.session_state.page == 'user_guide':
        guide_container.markdown(
            '<div style="height: 4px; width: 100%; background-color: #2196F3; border-radius: 2px; margin-bottom: 0.5rem;"></div>',
            unsafe_allow_html=True
        )
    guide_container.button("📖 User Guide", 
                     key="user_guide", 
                     on_click=lambda: set_page('user_guide'), 
                     use_container_width=True)
    
    # Add a spacer before logout
    st.sidebar.markdown("<hr style='margin: 0.5rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
    
    # Logout button
    logout_container = st.sidebar.container()
    logout_container.button("🚪 Logout", 
                     key="logout_button", 
                     on_click=logout_user, 
                     use_container_width=True)
    
    # Main navigation
    st.sidebar.header("Navigation")
    
    # Learning Dashboard button with highlight if current page
    dashboard_container = st.sidebar.container()
    if st.session_state.page == 'dashboard':
        dashboard_container.markdown(
            '<div style="height: 4px; width: 100%; background-color: #2196F3; border-radius: 2px; margin-bottom: 0.5rem;"></div>',
            unsafe_allow_html=True
        )
    dashboard_container.button("🏠 Learning Dashboard", 
                     on_click=lambda: set_page('dashboard'), 
                     use_container_width=True)
    
    # Learning Progress button with progress indication
    if not st.session_state.is_guest:
        total_modules = len(modules)
        overall_progress = sum(st.session_state.module_progress.values()) / total_modules if st.session_state.module_progress else 0
        
        # Progress button with highlight if current page
        progress_container = st.sidebar.container()
        if st.session_state.page == 'progress':
            progress_container.markdown(
                '<div style="height: 4px; width: 100%; background-color: #2196F3; border-radius: 2px; margin-bottom: 0.5rem;"></div>',
                unsafe_allow_html=True
            )
        progress_container.button(f"📊 Learning Progress [{int(overall_progress*100)}%]", 
                         on_click=lambda: set_page('progress'), 
                         use_container_width=True)
    
    # Modules navigation
    st.sidebar.header("Learning Modules")
    
    for i, module in enumerate(modules):
        # Get module progress
        progress = st.session_state.module_progress.get(i, 0)
        
        # Display progress indicator and button
        if progress > 0:
            # Use a color for completed/in-progress modules
            button_color = "#4CAF50" if progress >= 1.0 else "#FF9800"
            progress_text = f"[{int(progress*100)}%] "
            button_label = f"{i+1}. {module['name']}"
            full_label = f"{progress_text}{button_label}"
            
            # Create a container for the button to customize it further
            button_container = st.sidebar.container()
            button_container.markdown(
                f'<div style="margin-bottom: 0.5rem;"><div style="height: 4px; width: {int(progress*100)}%; background-color: {button_color}; border-radius: 2px;"></div></div>',
                unsafe_allow_html=True
            )
            button_container.button(full_label, 
                           on_click=lambda i=i: set_page_with_module('module', i), 
                           use_container_width=True,
                           help=f"Module {i+1}: {module['name']} - {int(progress*100)}% complete")
        else:
            st.sidebar.button(f"{i+1}. {module['name']}", 
                           on_click=lambda i=i: set_page_with_module('module', i), 
                           use_container_width=True)

    # Practical Examples button with enhanced styling
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    examples_container = st.sidebar.container()
    
    # Custom styling for the Practical Examples button to make it stand out
    examples_container.markdown("""
    <style>
    .examples-button {
        background: linear-gradient(90deg, #FF5722, #FF9800);
        color: white;
        font-weight: bold;
        text-align: center;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        cursor: pointer;
        box-shadow: 0 3px 8px rgba(255, 87, 34, 0.3);
        transition: all 0.3s;
        border: none;
        font-size: 1.1rem;
        display: block;
        width: 100%;
    }
    .examples-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 87, 34, 0.4);
    }
    .examples-header {
        padding-top: 10px;
        margin-bottom: 5px;
        font-size: 1.1rem;
        color: #555;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add a nice header for the examples section
    examples_container.markdown('<div class="examples-header">Try our</div>', unsafe_allow_html=True)
    
    # Add highlight if this is the current page
    if st.session_state.page == 'examples':
        examples_container.markdown(
            '<div style="height: 4px; width: 100%; background-color: #2196F3; border-radius: 2px; margin-bottom: 0.5rem;"></div>',
            unsafe_allow_html=True
        )
    
    # Создаем красивую кнопку с помощью CSS, но делаем ее простым HTML-элементом без JavaScript
    examples_container.markdown("""
    <style>
    .examples-button {
        display: inline-block;
        background: linear-gradient(90deg, #FF5722, #FF9800);
        color: white;
        font-weight: bold;
        padding: 12px 20px;
        border-radius: 8px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(255, 87, 34, 0.3);
        transition: all 0.3s;
        border: none;
        cursor: pointer;
        width: 100%;
        font-size: 1.1rem;
        text-decoration: none;
    }
    .examples-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 87, 34, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

    # Стили для кнопки - делаем текст жирным и отцентрированным
    examples_container.markdown("""
    <style>
    /* Стилизуем кнопку примеров ML */
    [data-testid="stButton"] > button {
        font-weight: 900 !important;
        text-align: center !important;
        letter-spacing: 0.5px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Используем стандартную кнопку Streamlit, но добавляем класс для стилизации
    if examples_container.button("🧪 Practical ML Examples", key="examples_button", use_container_width=True):
        set_page('examples')
        st.rerun()  # Перезапускаем приложение, чтобы отразить изменение
        
    # Добавляем CSS для стилизации стандартной кнопки Streamlit
    examples_container.markdown("""
    <style>
    /* Стилизуем стандартную кнопку Streamlit */
    [data-testid="stButton"] > button {
        background: linear-gradient(90deg, #FF5722, #FF9800);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        transition: all 0.3s;
    }
    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 87, 34, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show the selected page content
    if st.session_state.page == 'dashboard':
        show_dashboard()
    elif st.session_state.page == 'module':
        show_module(st.session_state.current_module)
    elif st.session_state.page == 'progress':
        show_learning_progress()
    elif st.session_state.page == 'examples':
        show_practical_examples()
    elif st.session_state.page == 'user_guide':
        show_user_guide()

def show_dashboard():
    # Custom header with enhanced fire icons and decorative elements
    st.markdown("""
    <style>
    .dashboard-header {
        text-align: center;
        position: relative;
        padding: 25px 0 15px;
        margin-bottom: 30px;
        background: linear-gradient(135deg, #FFF8F0 0%, #FFE8E0 100%);
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(255, 87, 34, 0.1);
        border: 1px solid rgba(255, 87, 34, 0.2);
    }
    .fire-icon {
        font-size: 32px;
        display: inline-block;
        margin: 0 15px;
        animation: flicker 1.5s infinite alternate;
        filter: drop-shadow(0 2px 4px rgba(255, 87, 34, 0.3));
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF5722, #FF9800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0 5px;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.05);
    }
    .decorative-line {
        height: 4px;
        background: linear-gradient(90deg, rgba(255,87,34,0) 0%, rgba(255,87,34,1) 50%, rgba(255,87,34,0) 100%);
        margin: 15px auto;
        width: 80%;
        border-radius: 2px;
    }
    .flame-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 10px;
        position: relative;
    }
    .header-subtitle {
        font-size: 1.3rem;
        margin-top: 10px;
        color: #666;
        font-weight: 500;
    }
    .dashboard-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 4px solid #FF5722;
    }
    .stat-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9F9F9 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.3s;
        border: 1px solid #eee;
    }
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #FF5722;
        margin: 5px 0;
    }
    .stat-label {
        font-size: 1rem;
        color: #666;
        font-weight: 500;
    }
    .module-progress-container {
        margin: 20px 0;
    }
    .module-card {
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.2s;
        border: 1px solid #eee;
    }
    .module-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .module-title {
        font-weight: 600;
        margin-bottom: 5px;
        color: #333;
        display: flex;
        justify-content: space-between;
    }
    .module-stats {
        display: flex;
        justify-content: space-between;
        margin-top: 8px;
        font-size: 0.9rem;
        color: #666;
    }
    .progress-container {
        width: 100%;
        background-color: #f1f1f1;
        border-radius: 10px;
        height: 8px;
        margin: 5px 0;
        overflow: hidden;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s;
    }
    .continue-btn {
        display: block;
        background: linear-gradient(90deg, #FF5722, #FF9800);
        color: white;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        text-align: center;
        margin: 20px auto;
        box-shadow: 0 4px 12px rgba(255, 87, 34, 0.3);
        transition: all 0.3s;
        border: none;
        cursor: pointer;
        width: 100%;
        font-size: 1.1rem;
        text-decoration: none;
    }
    .continue-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 87, 34, 0.4);
    }
    .stats-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .stats-icon {
        font-size: 1.4rem;
        margin-right: 10px;
        color: #FF5722;
    }
    .time-chart-container {
        padding: 15px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-top: 20px;
        border: 1px solid #eee;
    }
    @keyframes flicker {
        0%, 100% { opacity: 1; transform: scale(1.0); }
        25% { opacity: 0.85; transform: scale(0.98); }
        50% { opacity: 0.9; transform: scale(1.02); }
        75% { opacity: 0.8; transform: scale(0.95); }
    }
    </style>
    
    <div class="dashboard-header">
        <div class="flame-container">
            <span class="fire-icon">🔥</span>
            <span class="fire-icon" style="animation-delay: 0.5s;">🔥</span>
            <span class="fire-icon" style="animation-delay: 0.3s;">🔥</span>
            <span class="fire-icon" style="animation-delay: 0.7s;">🔥</span>
        </div>
        <div class="decorative-line"></div>
        <div class="header-title">ML FireSafetyTutor</div>
        <div class="decorative-line"></div>
        <p class="header-subtitle">Your Interactive Learning Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get user data for detailed stats
    total_modules = len(modules)
    completed_modules = sum(1 for progress in st.session_state.module_progress.values() if progress >= 1.0)
    in_progress_modules = sum(1 for progress in st.session_state.module_progress.values() if 0 < progress < 1.0)
    not_started_modules = total_modules - completed_modules - in_progress_modules
    overall_progress = sum(st.session_state.module_progress.values()) / total_modules if st.session_state.module_progress else 0
    
    # Enhanced progress metrics in nice cards
    st.markdown('<div class="stats-header"><span class="stats-icon">📊</span><h2>Learning Statistics</h2></div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{int(overall_progress*100)}%</div>
            <div class="stat-label">Overall Completion</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{completed_modules}</div>
            <div class="stat-label">Completed Modules</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{in_progress_modules}</div>
            <div class="stat-label">In Progress</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{not_started_modules}</div>
            <div class="stat-label">Not Started</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Study time data (we'll simulate this based on progress)
    total_study_time = 0
    module_times = {}
    for i, module in enumerate(modules):
        progress = st.session_state.module_progress.get(i, 0)
        # Simulate module study time based on progress
        if progress > 0:
            est_module_time = min(180, max(30, int(progress * 150)))  # Between 30 and 150 minutes
            total_study_time += est_module_time
            module_times[i] = est_module_time
    
    study_time_hours = total_study_time // 60
    study_time_mins = total_study_time % 60
    
    # Display the total study time if user has studied
    if total_study_time > 0:
        st.markdown(f"""
        <div class="dashboard-card">
            <h3>📚 Study Time Summary</h3>
            <p>You've spent <span style="font-weight: bold; color: #FF5722;">{study_time_hours} hours and {study_time_mins} minutes</span> learning ML for Fire Safety!</p>
            <p>Keep up the great work! Consistent study habits lead to better retention and mastery.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced progress bars for each module
    st.markdown('<div class="stats-header"><span class="stats-icon">📚</span><h2>Module Progress</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="module-progress-container">', unsafe_allow_html=True)
    for i, module in enumerate(modules):
        progress = st.session_state.module_progress.get(i, 0)
        progress_pct = int(progress * 100)
        
        # Determine color based on progress
        color = "#FF9800"  # Default orange (in progress)
        status = "In Progress"
        if progress >= 1.0:
            color = "#4CAF50"  # Green (completed)
            status = "Completed"
        elif progress == 0:
            color = "#9E9E9E"  # Gray (not started)
            status = "Not Started"
        
        # Calculate time spent (simulated)
        time_spent = module_times.get(i, 0)
        time_display = f"{time_spent} min" if time_spent < 60 else f"{time_spent//60}h {time_spent%60}m"
        
        # Determine icon based on progress
        icon = "✅" if progress >= 1.0 else "🔄" if progress > 0 else "⬜"
        
        st.markdown(f"""
        <div class="module-card">
            <div class="module-title">
                <span>{icon} Module {i+1}: {module['name']}</span>
                <span style="color: {color};">{progress_pct}%</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {progress_pct}%; background-color: {color};"></div>
            </div>
            <div class="module-stats">
                <span>Status: <b style="color: {color};">{status}</b></span>
                <span>Time: {time_display}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Continue learning button with enhanced design
    st.markdown('<div class="stats-header"><span class="stats-icon">🔥</span><h2>Continue Your Learning Journey</h2></div>', unsafe_allow_html=True)
    
    # Find the next incomplete module
    next_module = 0
    for i in range(total_modules):
        if st.session_state.module_progress.get(i, 0) < 1.0:
            next_module = i
            break
    
    # Используем прямую кнопку Streamlit без JavaScript
    # Кнопка для продолжения изучения модуля
    continue_button = st.button(f"Continue with Module {next_module+1}: {modules[next_module]['name']}", 
                key=f"continue_module_{next_module}",
                on_click=lambda: set_page_with_module('module', next_module))
                
    # Добавляем CSS для стилизации кнопки продолжения
    st.markdown("""
    <style>
    /* Стилизуем кнопку продолжения обучения */
    [data-testid="stButton"] > button {
        background: linear-gradient(90deg, #FF5722, #FF9800);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1rem;
        font-size: 1.1rem;
        transition: all 0.3s;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    [data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 87, 34, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

def show_module(module_index):
    module = modules[module_index]
    
    # Track time spent on this module
    if not st.session_state.is_guest and st.session_state.authenticated:
        # Start tracking if not already tracking this module
        if not hasattr(st.session_state, 'current_tracking') or \
           st.session_state.current_tracking.get('module_id') != module_index:
            
            # If switching from another module, save the time spent on previous module
            if hasattr(st.session_state, 'current_tracking') and \
               'start_time' in st.session_state.current_tracking:
                prev_module = st.session_state.current_tracking.get('module_id')
                elapsed_time = (time.time() - st.session_state.current_tracking['start_time']) / 60  # Convert to minutes
                
                # Only update if significant time was spent (more than 1 minute)
                if elapsed_time > 1:
                    update_user_progress(
                        st.session_state.user_id,
                        prev_module,
                        st.session_state.module_progress.get(prev_module, 0),
                        time_spent=int(elapsed_time)
                    )
            
            # Start tracking current module
            st.session_state.current_tracking = {
                'module_id': module_index,
                'start_time': time.time(),
                'visits': 1
            }
            
            # Increment visits count
            if not st.session_state.is_guest:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('''
                UPDATE user_progress 
                SET visits_count = visits_count + 1
                WHERE user_id = %s AND module_id = %s
                ''', (st.session_state.user_id, module_index))
                conn.commit()
                conn.close()
    
    # Header with module info
    st.title(f"Module {module_index + 1}: {module['name']}")
    
    # Display module content
    module['function'].show_module_content()
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if module_index > 0:
            st.button("← Previous Module", 
                    on_click=lambda: set_page_with_module('module', module_index - 1), 
                    use_container_width=True)
    
    with col3:
        if module_index < len(modules) - 1:
            st.button("Next Module →", 
                    on_click=lambda: set_page_with_module('module', module_index + 1), 
                    use_container_width=True)
    
    with col2:
        st.button("Return to Learning Dashboard", on_click=lambda: set_page('dashboard'), use_container_width=True)

def show_learning_progress():
    st.title("📊 Your Learning Progress")
    
    # Style container
    st.markdown("""
    <style>
    .progress-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #4CAF50;
    }
    .stats-card {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .time-stat {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .time-icon {
        font-size: 24px;
        margin-right: 10px;
        color: #2196F3;
    }
    .quiz-score {
        font-size: 18px;
        font-weight: bold;
    }
    .stat-value {
        font-weight: bold;
        font-size: 24px;
    }
    .module-detail-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .detail-header {
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 1px solid #eee;
    }
    .detail-stat {
        display: inline-block;
        margin-right: 15px;
        color: #555;
    }
    .detail-progress-bar {
        height: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        margin-top: 8px;
        margin-bottom: 15px;
    }
    .progress-fill {
        height: 100%;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Get detailed progress info
    module_details = st.session_state.module_progress_details if 'module_progress_details' in st.session_state else {}
    
    # Overall stats
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.header("Overall Learning Statistics")
    
    total_modules = len(modules)
    completed_modules = sum(1 for progress in st.session_state.module_progress.values() if progress >= 1.0)
    in_progress_modules = sum(1 for progress in st.session_state.module_progress.values() if 0 < progress < 1.0)
    not_started_modules = total_modules - completed_modules - in_progress_modules
    overall_progress = sum(st.session_state.module_progress.values()) / total_modules if st.session_state.module_progress else 0
    
    # Calculate total time spent across all modules
    total_time_spent = sum(details.get('time_spent_minutes', 0) for details in module_details.values())
    total_visits = sum(details.get('visits_count', 0) for details in module_details.values())
    
    # Row 1: Summary metrics in colorful cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="stats-card" style="border-top: 4px solid #2196F3;">', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#2196F3;">{int(overall_progress*100)}%</h1>', unsafe_allow_html=True)
        st.markdown('<p>Overall Progress</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="stats-card" style="border-top: 4px solid #4CAF50;">', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#4CAF50;">{completed_modules}</h1>', unsafe_allow_html=True)
        st.markdown('<p>Completed Modules</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="stats-card" style="border-top: 4px solid #FF9800;">', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#FF9800;">{in_progress_modules}</h1>', unsafe_allow_html=True)
        st.markdown('<p>In Progress</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="stats-card" style="border-top: 4px solid #F44336;">', unsafe_allow_html=True)
        st.markdown(f'<h1 style="color:#F44336;">{not_started_modules}</h1>', unsafe_allow_html=True)
        st.markdown('<p>Not Started</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Row 2: Time metrics
    st.subheader("Time & Engagement Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hours = total_time_spent // 60
        minutes = total_time_spent % 60
        time_display = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        
        st.markdown('<div class="time-stat">', unsafe_allow_html=True)
        st.markdown('<span class="time-icon">⏱️</span>', unsafe_allow_html=True)
        st.markdown(f'<div><span class="stat-value">{time_display}</span><br>Total Learning Time</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        avg_time_per_module = round(total_time_spent / max(1, completed_modules + in_progress_modules))
        
        st.markdown('<div class="time-stat">', unsafe_allow_html=True)
        st.markdown('<span class="time-icon">🔄</span>', unsafe_allow_html=True)
        st.markdown(f'<div><span class="stat-value">{total_visits}</span><br>Total Module Visits</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Get test results if available
    test_results = get_user_test_results(st.session_state.user_id) if 'user_id' in st.session_state else []
    
    if test_results:
        # Average quiz score
        avg_score = sum(result['score'] for result in test_results) / len(test_results)
        
        st.markdown('<div style="margin-top: 15px;">', unsafe_allow_html=True)
        st.markdown(f'<span class="quiz-score">Average Quiz Score: {avg_score:.1f}%</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed module progress
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.header("Detailed Module Progress")
    
    # Module progress charts
    st.subheader("Progress by Module")
    
    # Completion tab and time spent tab
    tabs = st.tabs(["Completion Percentage", "Time Spent", "Module Visits"])
    
    with tabs[0]:
        # Module progress chart
        module_names = [f"Module {i+1}" for i in range(total_modules)]
        module_progress_values = [st.session_state.module_progress.get(i, 0) * 100 for i in range(total_modules)]
        
        # Create figure with improved styling
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=module_names,
            y=module_progress_values,
            marker_color=[
                FIRESAFETY_COLORS['secondary'] if p >= 100 else 
                FIRESAFETY_COLORS['warning'] if p > 0 else 
                FIRESAFETY_COLORS['error'] for p in module_progress_values
            ],
            text=[f"{p:.0f}%" for p in module_progress_values],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Completion: %{y:.1f}%<extra></extra>'
        ))
        
        # Apply the theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title="Module Completion Progress",
            height=450
        )
        
        # Additional customizations specific to this chart
        fig.update_layout(
            xaxis_title="Modules",
            yaxis_title="Completion Percentage",
            yaxis=dict(range=[0, 105]),  # Add a little padding above 100%
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        # Time spent chart
        time_spent_values = [module_details.get(i, {}).get('time_spent_minutes', 0) for i in range(total_modules)]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=module_names,
            y=time_spent_values,
            marker_color=FIRESAFETY_COLORS['tertiary'],
            text=[f"{t} min" for t in time_spent_values],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Time Spent: %{y} minutes<extra></extra>'
        ))
        
        # Apply the theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title="Time Spent on Each Module",
            height=450
        )
        
        # Additional customizations
        fig.update_layout(
            xaxis_title="Modules",
            yaxis_title="Time (minutes)",
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        # Module visits chart
        visits_count_values = [module_details.get(i, {}).get('visits_count', 0) for i in range(total_modules)]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=module_names,
            y=visits_count_values,
            marker_color=FIRESAFETY_SEQUENTIAL[3],
            text=[f"{v}" for v in visits_count_values],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Visits: %{y}<extra></extra>'
        ))
        
        # Apply the theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title="Number of Times Each Module Was Accessed",
            height=450
        )
        
        # Additional customizations
        fig.update_layout(
            xaxis_title="Modules",
            yaxis_title="Number of Visits",
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed module cards
    st.subheader("Module Details")
    
    # Create module cards with detailed information
    for i, module in enumerate(modules):
        progress = st.session_state.module_progress.get(i, 0)
        progress_percentage = int(progress * 100)
        status = "Completed" if progress >= 1.0 else "In Progress" if progress > 0 else "Not Started"
        
        # Get additional details
        time_spent = module_details.get(i, {}).get('time_spent_minutes', 0)
        visits = module_details.get(i, {}).get('visits_count', 0)
        last_accessed = module_details.get(i, {}).get('last_accessed', None)
        
        # Format last accessed date
        last_accessed_str = ""
        if last_accessed:
            if isinstance(last_accessed, str):
                last_accessed_str = last_accessed.split('.')[0].replace('T', ' ')
            else:
                last_accessed_str = last_accessed.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get test results for this module
        module_test_results = [r for r in test_results if r['module_id'] == i]
        best_score = max([r['score'] for r in module_test_results], default=0)
        
        # Create the detail card
        st.markdown(f'<div class="module-detail-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="detail-header"><h3>Module {i+1}: {module["name"]}</h3></div>', unsafe_allow_html=True)
        
        # Progress bar
        progress_color = "#4CAF50" if progress >= 1.0 else "#FF9800" if progress > 0 else "#F44336"
        st.markdown(f'''
            <div>Progress: <strong>{progress_percentage}%</strong> ({status})</div>
            <div class="detail-progress-bar">
                <div class="progress-fill" style="width: {progress_percentage}%; background-color: {progress_color};"></div>
            </div>
        ''', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f'<div class="detail-stat">⏱️ Time Spent: <strong>{time_spent} min</strong></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="detail-stat">🔄 Visits: <strong>{visits}</strong></div>', unsafe_allow_html=True)
        
        with col3:
            if best_score > 0:
                st.markdown(f'<div class="detail-stat">📝 Best Quiz Score: <strong>{best_score:.1f}%</strong></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="detail-stat">📝 Quiz: <strong>Not taken</strong></div>', unsafe_allow_html=True)
        
        if last_accessed_str:
            st.markdown(f'<div style="margin-top: 10px; color: #666;">Last accessed: {last_accessed_str}</div>', unsafe_allow_html=True)
        
        # Action button
        if st.button(f"Go to Module {i+1}", key=f"goto_module_{i}"):
            set_page_with_module('module', i)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.header("Recommendations")
    
    # Find the next incomplete module
    next_module = 0
    for i in range(total_modules):
        if st.session_state.module_progress.get(i, 0) < 1.0:
            next_module = i
            break
    
    st.write(f"We recommend continuing with **Module {next_module+1}: {modules[next_module]['name']}**")
    
    # Continue learning button
    st.button("Continue Learning", 
             on_click=lambda: set_page_with_module('module', next_module), 
             use_container_width=True,
             key="continue_learning_main")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_user_guide():
    st.title("📖 ML FireSafetyTutor User Guide")
    
    # Load and display user guide content
    with open('user_guide.html', 'r') as file:
        user_guide_html = file.read()
    
    st.components.v1.html(user_guide_html, height=600, scrolling=True)
    
    # Return to learning dashboard button
    st.button("Return to Learning Dashboard", on_click=lambda: set_page('dashboard'), use_container_width=True)

def show_practical_examples():
    st.title("🔬 Practical ML Examples in Fire Safety")
    
    # Style container
    st.markdown("""
    <style>
    .example-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .example-1 { border-left: 4px solid #FF5722; }
    .example-2 { border-left: 4px solid #2196F3; }
    .example-3 { border-left: 4px solid #4CAF50; }
    .example-4 { border-left: 4px solid #9C27B0; }
    .example-5 { border-left: 4px solid #FF9800; }
    </style>
    """, unsafe_allow_html=True)
    
    st.write("""
    These practical examples demonstrate real-world applications of machine learning in fire safety. 
    Explore each example to understand how ML techniques are applied to solve critical fire safety challenges.
    """)
    
    # Example tabs
    example_tabs = st.tabs([
        "1. Fire Risk Prediction", 
        "2. Thermal Hotspot Detection", 
        "3. Evacuation Planning", 
        "4. Maintenance Prediction", 
        "5. Fire Spread Simulation"
    ])
    
    # Example 1: Fire Risk Prediction
    with example_tabs[0]:
        st.markdown('<div class="example-container example-1">', unsafe_allow_html=True)
        st.header("Fire Risk Prediction for Buildings")
        
        st.write("""
        This example demonstrates how machine learning can predict fire risk levels for different buildings
        based on various structural and environmental factors.
        """)
        
        # Interactive inputs
        st.subheader("Try the Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            building_age = st.slider("Building Age (years)", 1, 100, 25)
            floor_area = st.slider("Floor Area (sq. meters)", 50, 5000, 1000)
            occupancy = st.number_input("Daily Occupancy (people)", 1, 1000, 50)
        
        with col2:
            has_sprinklers = st.checkbox("Sprinkler System Installed", True)
            has_alarm = st.checkbox("Fire Alarm System Installed", True)
            has_extinguishers = st.checkbox("Fire Extinguishers Available", True)
            
        building_type = st.selectbox("Building Type", ["Residential", "Commercial", "Industrial", "Educational", "Healthcare"])
        
        # Simplified risk calculation logic
        risk_score = 0
        risk_score += building_age / 10  # Older buildings have higher risk
        risk_score += floor_area / 1000  # Larger buildings have higher risk
        risk_score += occupancy / 100    # More people means higher risk
        
        # Safety measures reduce risk
        if has_sprinklers:
            risk_score -= 2
        if has_alarm:
            risk_score -= 1
        if has_extinguishers:
            risk_score -= 0.5
            
        # Building type factors
        type_factors = {
            "Residential": 1.0,
            "Commercial": 1.2,
            "Industrial": 1.5,
            "Educational": 0.8,
            "Healthcare": 1.1
        }
        risk_score *= type_factors[building_type]
        
        # Normalize to 0-10 scale
        risk_score = max(0, min(10, risk_score))
        
        # Determine risk category
        if risk_score < 3:
            risk_category = "Low"
            risk_color = "#4CAF50"  # Green
        elif risk_score < 6:
            risk_category = "Medium"
            risk_color = "#FF9800"  # Orange
        else:
            risk_category = "High"
            risk_color = "#F44336"  # Red
            
        # Display prediction
        st.subheader("Risk Assessment Results")
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Fire Risk Score: {risk_category}"},
            number = {'font': {'color': risk_color, 'size': 40}},
            gauge = {
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': FIRESAFETY_COLORS['text']},
                'bar': {'color': risk_color},
                'bgcolor': FIRESAFETY_COLORS['background'],
                'borderwidth': 2,
                'bordercolor': FIRESAFETY_COLORS['light_text'],
                'steps': [
                    {'range': [0, 3], 'color': '#E8F5E9'},  # Light green
                    {'range': [3, 6], 'color': '#FFF3E0'},  # Light orange
                    {'range': [6, 10], 'color': '#FFEBEE'}  # Light red
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 2},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))
        
        # Apply our theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title="Fire Risk Assessment",
            height=320
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk factors explanation
        st.subheader("Key Risk Factors")
        
        factors_data = [
            {"Factor": "Building Age", "Impact": building_age / 10, "Weight": "High"},
            {"Factor": "Floor Area", "Impact": floor_area / 1000, "Weight": "Medium"},
            {"Factor": "Occupancy", "Impact": occupancy / 100, "Weight": "Medium"},
            {"Factor": "Building Type", "Impact": type_factors[building_type], "Weight": "High"},
            {"Factor": "Safety Systems", "Impact": -2 if has_sprinklers else 0, "Weight": "Very High"}
        ]
        
        factors_df = pd.DataFrame(factors_data)
        
        fig = px.bar(
            factors_df, 
            y="Factor", 
            x="Impact", 
            color="Impact",
            color_continuous_scale=FIRESAFETY_SEQUENTIAL,
            labels={"Impact": "Risk Contribution", "Factor": "Risk Factor"}
        )
        
        # Apply the unified theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title="Risk Factor Contribution Analysis",
            height=400
        )
        
        # Add hover information
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.2f}<br>Weight: %{customdata}<extra></extra>',
            customdata=factors_df['Weight']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("Recommendations")
        
        recommendations = []
        if building_age > 50:
            recommendations.append("Consider structural inspection due to building age")
        if not has_sprinklers:
            recommendations.append("Install automatic sprinkler system")
        if not has_alarm:
            recommendations.append("Install fire alarm system")
        if floor_area > 2000 and occupancy > 100:
            recommendations.append("Implement additional emergency exits")
            
        if not recommendations:
            recommendations.append("Maintain current fire safety measures")
            
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Example 2: Thermal Hotspot Detection
    with example_tabs[1]:
        st.markdown('<div class="example-container example-2">', unsafe_allow_html=True)
        st.header("Thermal Hotspot Detection for Fire Prevention")
        
        st.write("""
        This example demonstrates how machine learning algorithms can analyze thermal imagery to identify
        potential fire hazards before they ignite, enabling preventive intervention.
        """)
        
        # Create interactive thermal imaging simulation
        st.subheader("Interactive Thermal Analysis")
        
        # Create simulation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            environment = st.selectbox(
                "Environment Type", 
                ["Industrial Facility", "Office Building", "Electrical Substation", "Server Room"]
            )
            scan_resolution = st.select_slider(
                "Scan Resolution", 
                options=["Low", "Medium", "High"]
            )
        
        with col2:
            monitoring_duration = st.slider("Monitoring Duration (hours)", 1, 48, 24)
            temperature_threshold = st.slider("Alert Temperature Threshold (°C)", 60, 120, 85)
        
        # Generate simulated thermal data based on parameters
        np.random.seed(42)  # For reproducible results
        
        # Environment affects base temperature and variance
        env_temp_map = {
            "Industrial Facility": (45, 15),  # (base_temp, variance)
            "Office Building": (30, 10), 
            "Electrical Substation": (50, 20),
            "Server Room": (40, 12)
        }
        
        base_temp, variance = env_temp_map[environment]
        
        # Generate a grid of temperatures
        grid_size = {"Low": 10, "Medium": 15, "High": 20}[scan_resolution]
        
        # Create temperature grid with natural patterns and hotspots
        temp_grid = np.random.normal(base_temp, variance/3, (grid_size, grid_size))
        
        # Add realistic patterns
        x, y = np.mgrid[0:grid_size, 0:grid_size]
        
        # Add gradient effect
        gradient = (x + y) / (2 * grid_size)
        temp_grid += gradient * variance/2
        
        # Add hotspots based on environment
        num_hotspots = {
            "Industrial Facility": 3,
            "Office Building": 1,
            "Electrical Substation": 4,
            "Server Room": 2
        }[environment]
        
        for _ in range(num_hotspots):
            hx = np.random.randint(0, grid_size)
            hy = np.random.randint(0, grid_size)
            # Create a hotspot with exponential falloff
            for i in range(grid_size):
                for j in range(grid_size):
                    dist = np.sqrt((i - hx)**2 + (j - hy)**2)
                    falloff = np.exp(-dist/2) * variance
                    temp_grid[i, j] += falloff
                    
        # Find hotspots exceeding threshold
        hotspots = []
        for i in range(grid_size):
            for j in range(grid_size):
                if temp_grid[i, j] > temperature_threshold:
                    hotspots.append((i, j, temp_grid[i, j]))
        
        # Visualization with plotly
        fig = go.Figure()
        
        # Create heatmap
        fig.add_trace(go.Heatmap(
            z=temp_grid,
            x=list(range(grid_size)),
            y=list(range(grid_size)),
            colorscale='Inferno',
            showscale=True,
            colorbar=dict(title="Temperature (°C)"),
            hovertemplate='Position: (%{x}, %{y})<br>Temperature: %{z:.1f}°C<extra></extra>'
        ))
        
        # Mark hotspots
        if hotspots:
            fig.add_trace(go.Scatter(
                x=[hs[1] for hs in hotspots],
                y=[hs[0] for hs in hotspots],
                mode='markers',
                marker=dict(
                    symbol='circle-open',
                    size=15,
                    color='white',
                    line=dict(width=2)
                ),
                hovertemplate='Hotspot<br>Position: (%{x}, %{y})<br>Temperature: %{text:.1f}°C<extra></extra>',
                text=[hs[2] for hs in hotspots],
                name='Hotspots'
            ))
        
        # Apply the firesafety theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title=f"Thermal Scan: {environment}",
            height=500
        )
        
        # Customize layout
        fig.update_layout(
            xaxis=dict(title="X Position (m)"),
            yaxis=dict(title="Y Position (m)"),
        )
        
        # Show the visualization
        st.plotly_chart(fig, use_container_width=True)
        
        # Results summary
        st.subheader("Analysis Results")
        
        total_area = grid_size * grid_size  # in square meters for simplicity
        hotspot_count = len(hotspots)
        avg_temp = np.mean(temp_grid)
        max_temp = np.max(temp_grid)
        
        # Create metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Temp", f"{avg_temp:.1f}°C")
        
        with col2:
            st.metric("Maximum Temp", f"{max_temp:.1f}°C")
            
        with col3:
            st.metric("Hotspots Detected", hotspot_count)
            
        with col4:
            risk_level = "Low" if hotspot_count == 0 else "Medium" if hotspot_count < 3 else "High"
            risk_color = {"Low": "green", "Medium": "orange", "High": "red"}[risk_level]
            st.markdown(f'<div style="padding:10px; background-color:{risk_color}; color:white; border-radius:5px; text-align:center;"><h4>Risk Level: {risk_level}</h4></div>', unsafe_allow_html=True)
        
        # Hourly temperature trends
        st.subheader("Temperature Trend Analysis")
        
        # Generate time-based data
        hours = list(range(1, monitoring_duration + 1))
        
        # Create trend lines for average and maximum temperatures
        avg_temps = []
        max_temps = []
        num_hotspots_over_time = []
        
        # Set a seed for reproducibility
        np.random.seed(seed=42)
        
        # Baseline temperatures
        baseline_avg = avg_temp
        baseline_max = max_temp
        
        # Generate realistic fluctuations over time
        for i in range(monitoring_duration):
            time_factor = i / monitoring_duration  # Normalized time
            
            # Add realistic patterns: temperatures tend to rise during the day, fall at night
            day_cycle = np.sin(2 * np.pi * ((i % 24) / 24)) * variance / 4
            
            # Create slight upward trend for demonstration
            trend_factor = time_factor * variance / 3
            
            # Current values with noise
            curr_avg = baseline_avg + day_cycle + trend_factor + np.random.normal(0, variance/10)
            
            # Max tends to fluctuate more
            curr_max = baseline_max + day_cycle * 1.5 + trend_factor * 1.2 + np.random.normal(0, variance/6)
            
            # Number of hotspots follows max temperature pattern
            curr_hotspots = max(0, int((curr_max - temperature_threshold) / 5) + np.random.randint(-1, 2))
            
            avg_temps.append(curr_avg)
            max_temps.append(curr_max)
            num_hotspots_over_time.append(curr_hotspots)
        
        # Create the figure
        fig = go.Figure()
        
        # Add average temperature line
        fig.add_trace(go.Scatter(
            x=hours,
            y=avg_temps,
            mode='lines',
            name='Average Temperature',
            line=dict(color=FIRESAFETY_COLORS['tertiary'], width=2)
        ))
        
        # Add maximum temperature line
        fig.add_trace(go.Scatter(
            x=hours,
            y=max_temps,
            mode='lines',
            name='Maximum Temperature',
            line=dict(color=FIRESAFETY_COLORS['primary'], width=2)
        ))
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            x=hours,
            y=[temperature_threshold] * len(hours),
            mode='lines',
            name='Alert Threshold',
            line=dict(color=FIRESAFETY_COLORS['error'], width=2, dash='dash')
        ))
        
        # Apply theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title="Temperature Trends Over Monitoring Period",
            height=400
        )
        
        # Add axis titles
        fig.update_layout(
            xaxis_title="Time (hours)",
            yaxis_title="Temperature (°C)",
        )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Hotspot tracking
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=num_hotspots_over_time,
            mode='lines+markers',
            name='Hotspot Count',
            line=dict(color=FIRESAFETY_COLORS['secondary'], width=2),
            marker=dict(size=8)
        ))
        
        # Apply theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title="Hotspot Count Over Time",
            height=300
        )
        
        # Add axis titles
        fig.update_layout(
            xaxis_title="Time (hours)",
            yaxis_title="Number of Hotspots",
            yaxis=dict(rangemode='nonnegative')  # Always start y at 0 or higher
        )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # How it works
        st.subheader("How Thermal Hotspot Detection Works")
        
        st.write("""
        The thermal hotspot detection system identifies potential fire hazards by analyzing thermal imagery:
        
        1. **Thermal Imaging**: Specialized infrared cameras capture the temperature distribution across surfaces
        2. **Image Processing**: Algorithms process the thermal data to identify regions with elevated temperatures
        3. **Pattern Recognition**: Machine learning models identify abnormal temperature patterns indicative of potential hazards
        4. **Time Series Analysis**: Temperature trends are monitored over time to detect gradual increases that may indicate developing issues
        5. **Risk Assessment**: The system evaluates detected hotspots based on temperature, location, and time patterns
        
        This approach allows for the early detection of potential fire hazards before conventional smoke or fire detectors would activate, providing critical time for preventive intervention.
        """)
        
        # Application areas
        st.subheader("Key Application Areas")
        
        # Create pie chart of application areas
        applications = {
            "Electrical Systems": 35,
            "Manufacturing Equipment": 25,
            "Data Centers": 15,
            "HVAC Systems": 10,
            "Warehouses": 10,
            "Other": 5
        }
        
        # Create pie chart
        fig = px.pie(
            values=list(applications.values()),
            names=list(applications.keys()),
            color_discrete_sequence=FIRESAFETY_CATEGORICAL,
        )
        
        # Apply theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title="Thermal Monitoring Applications",
            height=400
        )
        
        # Show the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample code for temperature analysis
        st.subheader("Sample Implementation")
        
        code = '''
        import numpy as np
        import cv2
        from sklearn.cluster import KMeans
        
        def analyze_thermal_image(thermal_image_path, temp_threshold=85):
            """
            Analyze a thermal image to detect potential hotspots
            
            Parameters:
            - thermal_image_path: Path to thermal image file
            - temp_threshold: Temperature threshold in Celsius
            
            Returns:
            - hotspots: List of detected hotspot coordinates and temperatures
            - annotated_image: Image with hotspots marked
            """
            # Load thermal image
            # Note: In real applications, this would use a calibrated thermal camera API
            thermal_img = cv2.imread(thermal_image_path)
            
            # Convert to grayscale (represents temperature in this simulation)
            gray_thermal = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)
            
            # Apply color mapping for visualization
            colored_thermal = cv2.applyColorMap(gray_thermal, cv2.COLORMAP_INFERNO)
            
            # Convert grayscale values to temperatures (simplified mapping)
            # In real applications, this would use calibration data from the thermal camera
            temperatures = gray_thermal.astype(float) * 0.5  # Example: each gray level = 0.5°C
            
            # Find potential hotspots (pixels above threshold)
            hotspot_mask = temperatures > temp_threshold
            hotspot_pixels = np.where(hotspot_mask)
            
            # Group nearby hotspots using clustering
            if len(hotspot_pixels[0]) > 0:
                # Prepare data for clustering
                hotspot_points = np.column_stack([hotspot_pixels[1], hotspot_pixels[0]])  # x,y format
                
                # Determine number of clusters based on hotspot count
                n_clusters = min(8, len(hotspot_points))
                
                # Apply KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(hotspot_points)
                
                # Get cluster centers (these are our consolidated hotspots)
                centers = kmeans.cluster_centers_
                
                # Prepare result with coordinates and temperatures
                hotspots = []
                for center in centers:
                    x, y = int(center[0]), int(center[1])
                    temp = temperatures[y, x]
                    hotspots.append((x, y, temp))
                    
                    # Mark hotspots on the image
                    cv2.circle(colored_thermal, (x, y), 15, (255, 255, 255), 2)
                    cv2.putText(colored_thermal, f"{temp:.1f}°C", (x+10, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                hotspots = []
            
            return hotspots, colored_thermal
            
        # Example usage:
        hotspots, thermal_img = analyze_thermal_image("equipment_thermal_scan.jpg", temp_threshold=85)
        print(f"Detected {len(hotspots)} potential fire hazards.")
        
        # Implement time series analysis for trend detection
        def analyze_temperature_trends(temp_history, threshold=5.0):
            """
            Analyze temperature trends to detect gradual increases
            
            Parameters:
            - temp_history: Dictionary mapping timestamps to temperature readings
            - threshold: Threshold for temperature increase rate
            
            Returns:
            - anomalies: List of timestamps where abnormal increases were detected
            """
            timestamps = sorted(temp_history.keys())
            if len(timestamps) < 3:
                return []  # Need more data points
                
            anomalies = []
            for i in range(2, len(timestamps)):
                t1, t2, t3 = timestamps[i-2], timestamps[i-1], timestamps[i]
                temp1, temp2, temp3 = temp_history[t1], temp_history[t2], temp_history[t3]
                
                # Check for consistent increases
                if temp3 > temp2 > temp1:
                    # Calculate rate of increase
                    time_diff = (t3 - t1).total_seconds() / 3600  # hours
                    temp_diff = temp3 - temp1
                    rate = temp_diff / time_diff  # °C per hour
                    
                    if rate > threshold:
                        anomalies.append((t3, temp3, rate))
            
            return anomalies
        '''
        
        st.code(code, language="python")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Example 3: Evacuation Planning
    with example_tabs[2]:
        st.markdown('<div class="example-container example-3">', unsafe_allow_html=True)
        st.header("Optimal Evacuation Route Planning")
        
        # Custom CSS for enhanced visualization
        st.markdown("""
        <style>
        .evacuation-stats {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #4CAF50;
        }
        .route-info {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }
        .route-icon {
            font-size: 24px;
            margin-right: 10px;
            color: #2196F3;
        }
        .risk-level {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 15px;
            font-weight: bold;
            text-align: center;
            margin: 5px 0;
            width: 100%;
            color: white;
        }
        .risk-low {
            background-color: #4CAF50;
        }
        .risk-medium {
            background-color: #FF9800;
        }
        .risk-high {
            background-color: #F44336;
        }
        .step-number {
            display: inline-block;
            width: 30px;
            height: 30px;
            line-height: 30px;
            text-align: center;
            background-color: #FF5722;
            color: white;
            border-radius: 50%;
            margin-right: 10px;
            font-weight: bold;
        }
        .building-animation {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 15px 0;
        }
        .algorithm-selector {
            background-color: #f1f8e9;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #8bc34a;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Introduction
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("""
            <div style="font-size: 1.1em; line-height: 1.5;">
            This example demonstrates how <strong>reinforcement learning</strong> can be used to determine optimal evacuation
            routes in buildings during emergencies, adapting to changing conditions like blocked exits or smoke-filled corridors.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #FFF3E0; padding: 10px; border-radius: 5px; border-left: 4px solid #FF9800;">
            <strong>🔎 Application Domain:</strong><br>
            • High-rise buildings<br>
            • Shopping malls<br>
            • Hospitals<br>
            • Industrial facilities
            </div>
            """, unsafe_allow_html=True)
            
        # Building layout visualization
        st.subheader("🏢 Interactive Building Layout Simulation")
        
        # Building type selection
        building_type = st.radio(
            "Select Building Type:",
            ["Office Building", "Shopping Mall", "Industrial Facility"],
            horizontal=True
        )
        
        # Create building layout based on selected type
        if building_type == "Office Building":
            layout_width, layout_height = 9, 9
            layout = np.ones((layout_height, layout_width))
            
            # Outer walls
            layout[0, :] = 0  # Top wall
            layout[-1, :] = 0  # Bottom wall
            layout[:, 0] = 0  # Left wall
            layout[:, -1] = 0  # Right wall
            
            # Inner walls - office layout
            layout[2, 1:6] = 0  # Horizontal corridor wall
            layout[6, 2:8] = 0  # Horizontal corridor wall
            layout[1:7, 4] = 0  # Vertical wall with door
            layout[5, 4] = 1    # Door in vertical wall
            
            # Conference room
            layout[3:6, 6:8] = 0  # Room walls
            layout[4, 6] = 1      # Room door
            
            # Add exits
            exit_positions = [(layout_height-1, 2), (layout_height-1, 7)]
            
            # Add starting point
            start_position = (1, 2)
            
        elif building_type == "Shopping Mall":
            layout_width, layout_height = 10, 10
            layout = np.ones((layout_height, layout_width))
            
            # Outer walls
            layout[0, :] = 0  # Top wall
            layout[-1, :] = 0  # Bottom wall
            layout[:, 0] = 0  # Left wall
            layout[:, -1] = 0  # Right wall
            
            # Inner walls - mall layout
            layout[3, 1:9] = 0  # Top corridor
            layout[7, 1:9] = 0  # Bottom corridor
            layout[3:8, 3] = 0  # Left stores wall
            layout[3:8, 6] = 0  # Right stores wall
            
            # Entrances to stores
            layout[5, 3] = 1  # Left store entrance
            layout[5, 6] = 1  # Right store entrance
            
            # Add exits
            exit_positions = [(layout_height-1, 2), (layout_height-1, 8)]
            
            # Add starting point
            start_position = (2, 5)
            
        else:  # Industrial Facility
            layout_width, layout_height = 10, 10
            layout = np.ones((layout_height, layout_width))
            
            # Outer walls
            layout[0, :] = 0  # Top wall
            layout[-1, :] = 0  # Bottom wall
            layout[:, 0] = 0  # Left wall
            layout[:, -1] = 0  # Right wall
            
            # Inner walls - industrial layout
            layout[2, 1:8] = 0    # Upper area partition
            layout[2, 3] = 1      # Door in upper partition
            layout[4:8, 2] = 0    # Vertical partition
            layout[6, 2:9] = 0    # Lower horizontal partition
            layout[6, 5] = 1      # Door in lower partition
            
            # Machinery/hazard areas
            layout[3:5, 6:9] = 0  # Enclosed machinery room
            layout[4, 6] = 1      # Door to machinery
            
            # Add exits
            exit_positions = [(layout_height-1, 1), (layout_height-1, 8)]
            
            # Add starting point
            start_position = (1, 5)
        
        # Mark exits and starting point in layout
        for x, y in exit_positions:
            layout[x, y] = 2
            
        layout[start_position] = 3
        
        # Add fire/hazard with more options
        col1, col2 = st.columns(2)
        
        with col1:
            fire_position = st.selectbox(
                "Select fire/hazard location:",
                options=["No Hazard", "Near Exit 1", "Building Center", "Near Start Position", "Multiple Hazards"]
            )
        
        with col2:
            hazard_type = st.selectbox(
                "Select hazard type:",
                options=["Fire", "Smoke", "Chemical Spill", "Structural Damage"]
            )
        
        # Add hazards to layout based on selection
        if fire_position == "Near Exit 1":
            if building_type == "Office Building":
                layout[layout_height-2, 1:4] = 4
            elif building_type == "Shopping Mall":
                layout[layout_height-2, 1:4] = 4
                layout[layout_height-3, 2] = 4
            else:
                layout[layout_height-2, 1] = 4
                layout[layout_height-3, 1] = 4
                layout[layout_height-2, 2] = 4
        elif fire_position == "Building Center":
            if building_type == "Office Building":
                layout[4, 3:6] = 4
                layout[5, 4] = 4
            elif building_type == "Shopping Mall":
                layout[5, 4:7] = 4
                layout[4, 5] = 4
                layout[6, 5] = 4
            else:
                layout[4, 4:7] = 4
                layout[5, 5] = 4
        elif fire_position == "Near Start Position":
            if building_type == "Office Building":
                layout[1, 3] = 4
                layout[2, 2] = 4
                layout[2, 3] = 4
            elif building_type == "Shopping Mall":
                layout[2, 4:7] = 4
                layout[1, 5] = 4
            else:
                layout[2, 4:7] = 4
                layout[1, 5] = 4
        elif fire_position == "Multiple Hazards":
            if building_type == "Office Building":
                layout[layout_height-2, 2] = 4  # Near exit
                layout[3, 2] = 4                # Mid building
                layout[1, 5] = 4                # Far corner
            elif building_type == "Shopping Mall":
                layout[layout_height-2, 2] = 4  # Near exit
                layout[5, 5] = 4                # Center
                layout[2, 8] = 4                # Upper right
            else:
                layout[layout_height-2, 1] = 4  # Near exit
                layout[3, 3] = 4                # Upper left
                layout[5, 8] = 4                # Right side
                
        # Hazard icon based on type
        hazard_icon = "🔥" if hazard_type == "Fire" else "💨" if hazard_type == "Smoke" else "☣️" if hazard_type == "Chemical Spill" else "💥"
        
        # Create a visualization container with border
        st.markdown('<div class="building-animation">', unsafe_allow_html=True)
        
        # Create layout visualization
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Apply fire safety theme to matplotlib
        fig, ax = apply_firesafety_theme_to_matplotlib(ax, f"{building_type} Layout", figsize=(12, 12))
        
        # Custom color scheme based on hazard type
        if hazard_type == "Fire":
            hazard_color = FIRESAFETY_COLORS['primary']  # Red for fire
        elif hazard_type == "Smoke":
            hazard_color = "#607D8B"  # Blue-grey for smoke
        elif hazard_type == "Chemical Spill":
            hazard_color = "#8BC34A"  # Green for chemical
        else:
            hazard_color = "#795548"  # Brown for structural damage
        
        # Use enhanced color scheme for better visualization
        colors = [
            FIRESAFETY_COLORS['text'],           # walls = black/dark
            FIRESAFETY_COLORS['background'],     # paths = white/light
            FIRESAFETY_COLORS['secondary'],      # exits = green
            FIRESAFETY_COLORS['tertiary'],       # start = blue
            hazard_color                         # hazard = varies by type
        ]
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        # Create the heatmap with improved styling
        heatmap = ax.imshow(layout, cmap=cmap, interpolation='nearest')
        
        # Add grid for clarity
        ax.set_xticks(np.arange(-.5, layout_width, 1), minor=True)
        ax.set_yticks(np.arange(-.5, layout_height, 1), minor=True)
        ax.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=1.5)
        
        # Remove axis labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add improved annotations with building-specific elements
        for i in range(layout_height):
            for j in range(layout_width):
                if layout[i, j] == 2:
                    ax.text(j, i, "EXIT", ha='center', va='center', color='white', 
                           fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                                                   fc=FIRESAFETY_COLORS['secondary'], 
                                                                   ec="white", lw=1, alpha=0.9))
                elif layout[i, j] == 3:
                    ax.text(j, i, "YOU", ha='center', va='center', color='white', 
                           fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                                                   fc=FIRESAFETY_COLORS['tertiary'], 
                                                                   ec="white", lw=1, alpha=0.9))
                elif layout[i, j] == 4:
                    ax.text(j, i, hazard_icon, ha='center', va='center', fontsize=18)
                
                # Add room labels for different building types
                if building_type == "Office Building":
                    if (i, j) == (1, 6):
                        ax.text(j, i, "Office", ha='center', va='center', color='#555', fontsize=8)
                    elif (i, j) == (4, 7):
                        ax.text(j, i, "Meeting\nRoom", ha='center', va='center', color='#555', fontsize=8)
                    elif (i, j) == (3, 2):
                        ax.text(j, i, "Office", ha='center', va='center', color='#555', fontsize=8)
                elif building_type == "Shopping Mall":
                    if (i, j) == (5, 2):
                        ax.text(j, i, "Store", ha='center', va='center', color='#555', fontsize=8)
                    elif (i, j) == (5, 8):
                        ax.text(j, i, "Store", ha='center', va='center', color='#555', fontsize=8)
                    elif (i, j) == (2, 5):
                        ax.text(j, i, "Food\nCourt", ha='center', va='center', color='#555', fontsize=8)
                elif building_type == "Industrial Facility":
                    if (i, j) == (1, 3):
                        ax.text(j, i, "Control\nRoom", ha='center', va='center', color='#555', fontsize=8)
                    elif (i, j) == (4, 7):
                        ax.text(j, i, "Machinery", ha='center', va='center', color='#555', fontsize=8)
                    elif (i, j) == (8, 5):
                        ax.text(j, i, "Storage", ha='center', va='center', color='#555', fontsize=8)
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', label='Walls', 
                      markerfacecolor=FIRESAFETY_COLORS['text'], markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Paths', 
                      markerfacecolor=FIRESAFETY_COLORS['background'], markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Exits', 
                      markerfacecolor=FIRESAFETY_COLORS['secondary'], markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Your Location', 
                      markerfacecolor=FIRESAFETY_COLORS['tertiary'], markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label=f'{hazard_type} Hazard', 
                      markerfacecolor=hazard_color, markersize=10),
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
                 ncol=5, frameon=True, fancybox=True, shadow=True)
        
        # Add a title with enhanced styling
        building_info = {
            "Office Building": "3-story Office Building (Floor 2)",
            "Shopping Mall": "Large Shopping Mall (Main Floor)",
            "Industrial Facility": "Manufacturing Plant (Production Floor)"
        }
        ax.set_title(f"{building_info[building_type]} - Emergency Layout", 
                    fontsize=16, color=FIRESAFETY_COLORS['text'], fontweight='bold', pad=20)
        
        # Display the visualization
        st.pyplot(fig)
        
        # End of building animation container
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Algorithm selection with enhanced styling
        st.markdown('<div class="algorithm-selector">', unsafe_allow_html=True)
        st.subheader("🧠 AI Evacuation Planning System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_algorithm = st.selectbox(
                "Select AI Routing Algorithm:",
                ["Q-Learning", "Deep Q-Network (DQN)", "A* Pathfinding", "Monte Carlo Tree Search"]
            )
            
        with col2:
            optimization_goal = st.selectbox(
                "Optimization Goal:",
                ["Shortest Path", "Lowest Risk", "Balanced (Time/Risk)", "Maximum Survivability"]
            )
        
        # Description of selected algorithm
        algorithm_descriptions = {
            "Q-Learning": "A reinforcement learning algorithm that learns the value of actions in states by learning a Q-function that assigns expected utility to state-action pairs.",
            "Deep Q-Network (DQN)": "An extension of Q-learning that uses neural networks to approximate the Q-function, allowing it to handle more complex environments.",
            "A* Pathfinding": "A best-first search algorithm that finds the shortest path between nodes using a heuristic function to guide the search.",
            "Monte Carlo Tree Search": "A heuristic search algorithm that builds a search tree through random sampling of the decision space."
        }
        
        st.info(f"**Algorithm Description**: {algorithm_descriptions[selected_algorithm]}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Evacuation route calculation with animation
        st.subheader("🚶 Optimal Evacuation Route Planning")
        
        # Add interactive parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            population_density = st.slider("Population Density", 0, 100, 50, 
                                        help="Higher density may impact evacuation speed and route selection")
        
        with col2:
            mobility_factor = st.slider("Mobility Constraints", 0, 100, 20, 
                                    help="Higher values represent more people with mobility limitations")
        
        with col3:
            panic_level = st.slider("Estimated Panic Level", 0, 100, 30, 
                                help="Higher panic levels may reduce optimal path-following")
        
        # Simulate computation with spinner
        with st.spinner(f"Computing optimal evacuation routes using {selected_algorithm}..."):
            time.sleep(0.5)  # Simulate processing time
            
            # Choose exit based on hazard location and building type
            target_exit = exit_positions[0]  # Default to first exit
            
            # Generate evacuation path based on selected options
            route = []
            current = start_position
            
            # Path generation logic based on hazard location
            if fire_position == "No Hazard":
                # Simple path logic - find closest exit
                if abs(start_position[0] - exit_positions[0][0]) + abs(start_position[1] - exit_positions[0][1]) > \
                   abs(start_position[0] - exit_positions[1][0]) + abs(start_position[1] - exit_positions[1][1]):
                    target_exit = exit_positions[1]
                else:
                    target_exit = exit_positions[0]
                    
            elif fire_position == "Near Exit 1":
                # Use second exit to avoid hazard
                target_exit = exit_positions[1]
                
            elif fire_position in ["Building Center", "Multiple Hazards"]:
                # Choose exit based on building type and optimization goal
                if optimization_goal == "Lowest Risk":
                    target_exit = exit_positions[1] if building_type == "Office Building" else exit_positions[0]
                else:
                    # For shortest path, determine nearest unobstructed exit
                    if abs(start_position[0] - exit_positions[0][0]) < abs(start_position[0] - exit_positions[1][0]):
                        target_exit = exit_positions[0]
                    else:
                        target_exit = exit_positions[1]
                
            else:  # Near Start Position
                # Move away from start to less hazardous exit
                if building_type == "Office Building":
                    target_exit = exit_positions[0]
                else:
                    target_exit = exit_positions[1]
            
            # Generate paths for different building types
            if building_type == "Office Building":
                # Generate different paths based on target exit
                if target_exit == exit_positions[0]:  # Exit on the left
                    if fire_position == "Near Start Position":
                        # Detour around fire
                        route = [start_position, (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), 
                                (6, 1), (7, 1), (7, 2), (8, 2)]
                    elif fire_position == "Building Center":
                        # Go around the center
                        route = [start_position, (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), 
                                (6, 1), (7, 1), (7, 2), (8, 2)]
                    else:
                        # Direct path to exit
                        route = [start_position, (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), 
                                (7, 2), (8, 2)]
                else:  # Exit on the right
                    if fire_position == "Near Start Position":
                        # Path avoiding the hazard
                        route = [start_position, (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), 
                                (1, 6), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), 
                                (6, 7), (7, 7), (8, 7)]
                    elif fire_position == "Building Center":
                        # Path avoiding center hazard
                        route = [start_position, (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), 
                                (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7)]
                    else:
                        # Direct path through office area
                        route = [start_position, (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
                                (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7), (8, 7)]
                                
            elif building_type == "Shopping Mall":
                # Generate different paths based on target exit
                if target_exit == exit_positions[0]:  # Left exit
                    if fire_position == "Multiple Hazards":
                        route = [start_position, (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), 
                                (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (8, 2), (9, 2)]
                    elif fire_position == "Building Center":
                        route = [start_position, (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), 
                                (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (8, 2), (9, 2)]
                    else:
                        route = [start_position, (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), 
                                (8, 5), (8, 4), (8, 3), (8, 2), (9, 2)]
                else:  # Right exit
                    if fire_position == "Near Start Position":
                        route = [start_position, (2, 4), (2, 3), (2, 2), (2, 1), (3, 1), 
                                (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (8, 2), (8, 3), 
                                (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (9, 8)]
                    else:
                        route = [start_position, (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), 
                                (8, 5), (8, 6), (8, 7), (8, 8), (9, 8)]
                                
            else:  # Industrial Facility
                # Generate different paths based on target exit
                if target_exit == exit_positions[0]:  # Left exit
                    if fire_position == "Near Start Position":
                        route = [start_position, (1, 4), (1, 3), (2, 3), (3, 3), (4, 3), 
                                (5, 3), (5, 2), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)]
                    else:
                        route = [start_position, (1, 4), (1, 3), (1, 2), (1, 1), (2, 1), 
                                (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 1)]
                else:  # Right exit
                    if fire_position == "Building Center":
                        route = [start_position, (1, 6), (1, 7), (1, 8), (2, 8), (3, 8), 
                                (4, 8), (5, 8), (6, 5), (7, 5), (8, 5), (8, 6), (8, 7), 
                                (8, 8), (9, 8)]
                    else:
                        route = [start_position, (1, 6), (1, 7), (1, 8), (2, 8), (3, 8), 
                                (4, 8), (5, 8), (6, 8), (7, 8), (8, 8), (9, 8)]
        
        # Create evacuation path visualization with animation
        path_layout = layout.copy()
        for x, y in route:
            # Don't override start, exit or fire cells
            if path_layout[x, y] not in [2, 3, 4]:
                path_layout[x, y] = 5
        
        # Create enhanced visualization for the route
        fig2, ax2 = plt.subplots(figsize=(12, 12))
        
        # Apply enhanced theme
        fig2, ax2 = apply_firesafety_theme_to_matplotlib(ax2, "AI-Generated Evacuation Route", figsize=(12, 12))
        
        # Extended color map with path
        route_color = FIRESAFETY_SEQUENTIAL[3]  # Default blue path
        if optimization_goal == "Lowest Risk":
            route_color = "#4CAF50"  # Green for safety-optimized
        elif optimization_goal == "Maximum Survivability":
            route_color = "#673AB7"  # Purple for survivability focus
            
        extended_colors = colors + [route_color]  # Add path color
        path_cmap = plt.matplotlib.colors.ListedColormap(extended_colors)
        
        heatmap2 = ax2.imshow(path_layout, cmap=path_cmap, interpolation='nearest')
        
        # Add grid for clarity
        ax2.set_xticks(np.arange(-.5, layout_width, 1), minor=True)
        ax2.set_yticks(np.arange(-.5, layout_height, 1), minor=True)
        ax2.grid(which='minor', color='#CCCCCC', linestyle='-', linewidth=1.5)
        
        # Remove axis labels
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Add enhanced legends with optimization goal information
        legend_title = f"AI-Optimized for: {optimization_goal}"
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', label='Walls', 
                     markerfacecolor=FIRESAFETY_COLORS['text'], markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Available Paths', 
                     markerfacecolor=FIRESAFETY_COLORS['background'], markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Emergency Exits', 
                     markerfacecolor=FIRESAFETY_COLORS['secondary'], markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='Your Location', 
                     markerfacecolor=FIRESAFETY_COLORS['tertiary'], markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label=f'{hazard_type} Hazard', 
                     markerfacecolor=hazard_color, markersize=10),
            plt.Line2D([0], [0], marker='s', color='w', label='AI-Generated Route', 
                     markerfacecolor=route_color, markersize=10),
        ]
        
        ax2.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
                  ncol=3, frameon=True, fancybox=True, shadow=True, title=legend_title)
        
        # Enhanced annotations
        for i in range(layout_height):
            for j in range(layout_width):
                if path_layout[i, j] == 2:
                    ax2.text(j, i, "EXIT", ha='center', va='center', color='white', 
                            fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                                                   fc=FIRESAFETY_COLORS['secondary'], 
                                                                   ec="white", lw=1, alpha=0.9))
                elif path_layout[i, j] == 3:
                    ax2.text(j, i, "YOU", ha='center', va='center', color='white', 
                            fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                                                   fc=FIRESAFETY_COLORS['tertiary'], 
                                                                   ec="white", lw=1, alpha=0.9))
                elif path_layout[i, j] == 4:
                    ax2.text(j, i, hazard_icon, ha='center', va='center', fontsize=18)
                elif path_layout[i, j] == 5:
                    # Show direction with arrows on path and number steps
                    if (i, j) in route[:-1]:  # Skip the last point
                        idx = route.index((i, j))
                        next_i, next_j = route[idx + 1]
                        
                        # Determine arrow direction
                        if next_i > i:
                            arrow = "↓"  # Down
                        elif next_i < i:
                            arrow = "↑"  # Up
                        elif next_j > j:
                            arrow = "→"  # Right
                        else:
                            arrow = "←"  # Left
                            
                        # Add step numbers on every third step for better readability
                        if idx % 3 == 0:
                            step_number = idx + 1
                            ax2.text(j, i, f"{step_number}", ha='center', va='center', color='white', 
                                fontsize=8, fontweight='bold', bbox=dict(boxstyle="circle", 
                                                                    fc=FIRESAFETY_COLORS['tertiary'], 
                                                                    ec="white", alpha=0.8))
                        else:
                            ax2.text(j, i, arrow, ha='center', va='center', color='black', 
                                fontweight='bold', fontsize=14)
        
        # Special marking for the destination exit
        i, j = target_exit
        ax2.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='white', linewidth=3, linestyle='--'))
        
        # Apply title
        algo_display_name = selected_algorithm
        if selected_algorithm == "A* Pathfinding":
            algo_display_name = "A* Algorithm"
        elif selected_algorithm == "Monte Carlo Tree Search":
            algo_display_name = "MCTS Algorithm"
            
        ax2.set_title(f"Optimal Evacuation Route \nGenerated with {algo_display_name}", 
                     fontsize=16, color=FIRESAFETY_COLORS['text'], fontweight='bold', pad=20)
        
        # Display the enhanced route visualization
        st.pyplot(fig2)
        
        # Enhanced evacuation statistics with styled container
        st.markdown('<div class="evacuation-stats">', unsafe_allow_html=True)
        st.subheader("📊 Evacuation Analytics & Statistics")
        
        # Calculate statistics
        path_length = len(route)
        avg_step_time = 1.5 + (mobility_factor / 100)  # Base time adjusted by mobility
        crowd_factor = 1 + (population_density / 200)  # Crowd slowing factor
        panic_factor = 1 + (panic_level / 200)        # Panic slowing factor
        
        estimated_time = path_length * avg_step_time * crowd_factor * panic_factor
        estimated_time = round(estimated_time, 1)
        
        # Risk calculations
        base_risk_score = 10  # Base risk
        
        # Increase risk based on hazards
        if fire_position == "No Hazard":
            risk_score = base_risk_score
            risk_level = "Low"
            risk_color = "risk-low"
        elif fire_position == "Near Exit 1":
            risk_score = base_risk_score + 10 + (panic_level / 10)
            risk_level = "Medium"
            risk_color = "risk-medium"
        elif fire_position == "Multiple Hazards":
            risk_score = base_risk_score + 25 + (panic_level / 5)
            risk_level = "High"
            risk_color = "risk-high" 
        else:
            risk_score = base_risk_score + 15 + (panic_level / 8)
            risk_level = "Medium"
            risk_color = "risk-medium"
            
        # Adjust risk based on algorithm and goal
        if optimization_goal == "Lowest Risk":
            risk_score = risk_score * 0.7  # Reduce risk
        elif optimization_goal == "Maximum Survivability":
            risk_score = risk_score * 0.8  # Somewhat reduce risk
            
        risk_score = round(risk_score, 1)
        
        # Mortality risk
        mortality_risk = risk_score / 4  # Convert to percentage, cap at 25%
        mortality_risk = min(25, mortality_risk)
        
        # Calculate survivability 
        survivability = 100 - mortality_risk
        
        # Statistics display with 3 columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="route-info">', unsafe_allow_html=True)
            st.markdown('<span class="route-icon">🛣️</span>', unsafe_allow_html=True)
            st.markdown(f'<div><span style="font-size: 24px; font-weight: bold;">{path_length}</span><br>Total Steps in Path</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="route-info">', unsafe_allow_html=True)
            st.markdown('<span class="route-icon">⏱️</span>', unsafe_allow_html=True)
            st.markdown(f'<div><span style="font-size: 24px; font-weight: bold;">{estimated_time}</span><br>Est. Evacuation Time (sec)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown(f'<div class="risk-level {risk_color}">Risk Level: {risk_level}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="margin-top: 5px;">Risk Score: {risk_score} / 50</div>', unsafe_allow_html=True)
            
        # Add advanced statistics
        st.markdown("<hr style='margin: 15px 0; opacity: 0.3;'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Show survivability with progress bar
            st.markdown(f'<div style="margin-bottom: 5px;"><strong>Survivability Rate:</strong> {survivability:.1f}%</div>', unsafe_allow_html=True)
            survivability_color = "#4CAF50" if survivability > 90 else "#FF9800" if survivability > 75 else "#F44336"
            st.markdown(f'''
                <div style="background-color: #e0e0e0; border-radius: 10px; height: 10px; width: 100%;">
                    <div style="background-color: {survivability_color}; width: {survivability}%; height: 100%; border-radius: 10px;"></div>
                </div>
            ''', unsafe_allow_html=True)
            
            # Environmental factors
            st.markdown("<br><strong>Environmental Influence Factors:</strong>", unsafe_allow_html=True)
            st.markdown(f"• Population Density Impact: +{(crowd_factor-1)*100:.1f}% evacuation time", unsafe_allow_html=True)
            st.markdown(f"• Mobility Constraints Impact: +{(avg_step_time-1.5)/1.5*100:.1f}% per step", unsafe_allow_html=True)
            
        with col2:
            # Route quality metrics
            score_percentage = 0
            if optimization_goal == "Shortest Path":
                # Higher score for shorter paths
                score_percentage = max(0, 100 - (path_length / 2))
            elif optimization_goal == "Lowest Risk":
                # Higher score for lower risk
                score_percentage = max(0, 100 - (risk_score * 2))
            elif optimization_goal == "Balanced (Time/Risk)":
                # Balance of time and risk
                score_percentage = max(0, 100 - (path_length / 4) - (risk_score))
            else:  # Maximum Survivability
                # Directly use survivability
                score_percentage = survivability
                
            score_percentage = min(99, score_percentage)  # Cap at 99%
            
            st.markdown(f'<div style="margin-bottom: 5px;"><strong>Route Quality Score:</strong> {score_percentage:.1f}%</div>', unsafe_allow_html=True)
            quality_color = "#4CAF50" if score_percentage > 75 else "#FF9800" if score_percentage > 50 else "#F44336"
            st.markdown(f'''
                <div style="background-color: #e0e0e0; border-radius: 10px; height: 10px; width: 100%;">
                    <div style="background-color: {quality_color}; width: {score_percentage}%; height: 100%; border-radius: 10px;"></div>
                </div>
            ''', unsafe_allow_html=True)
            
            # Behavioral factors
            st.markdown("<br><strong>Behavioral Influence Factors:</strong>", unsafe_allow_html=True)
            st.markdown(f"• Panic Level Impact: +{(panic_factor-1)*100:.1f}% evacuation time", unsafe_allow_html=True)
            st.markdown(f"• Crowding/Queuing at Exits: {'High' if population_density > 70 else 'Medium' if population_density > 40 else 'Low'}", unsafe_allow_html=True)
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # How it works section with enhanced styling
        st.subheader("🔍 How AI-Driven Evacuation Planning Works")
        
        # Steps with improved styling
        st.markdown("""
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-top: 15px;">
            <p><span class="step-number">1</span> <strong>Building Modeling:</strong> The AI system creates a detailed digital twin of the building including walls, doors, stairwells, and exit points.</p>
            
            <p><span class="step-number">2</span> <strong>Sensor Integration:</strong> Real-time data from smoke detectors, heat sensors, CCTV cameras, and occupancy counters is continuously fed into the system.</p>
            
            <p><span class="step-number">3</span> <strong>Hazard Modeling:</strong> The AI analyzes sensor data to identify hazards and predict their spread pattern over time, creating a dynamic risk map.</p>
            
            <p><span class="step-number">4</span> <strong>Path Computation:</strong> Using reinforcement learning algorithms, the system calculates optimal evacuation routes that minimize risk, distance, and congestion.</p>
            
            <p><span class="step-number">5</span> <strong>Continuous Update:</strong> As conditions change, the AI adapts evacuation routes in real-time and pushes updates to emergency signage and mobile devices.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Real-world applications
        st.markdown("""
        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; margin-top: 15px; border-left: 4px solid #4CAF50;">
            <h4 style="margin-top: 0;">Real-World Applications</h4>
            <ul style="margin-bottom: 0; padding-left: 20px;">
                <li><strong>Smart Buildings</strong> - Dynamic evacuation guidance through LED pathways and mobile alerts</li>
                <li><strong>Emergency Response Teams</strong> - AI-guided deployment to assist evacuations in high-risk areas</li>
                <li><strong>Public Venues</strong> - Optimized crowd management during emergencies in stadiums, theaters, and conference centers</li>
                <li><strong>Cruise Ships and Hotels</strong> - Personalized evacuation instructions for guests unfamiliar with the layout</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Example 4: Maintenance Prediction
    with example_tabs[3]:
        st.markdown('<div class="example-container example-4">', unsafe_allow_html=True)
        st.header("Fire Safety Equipment Maintenance Prediction")
        
        st.write("""
        This example shows how predictive maintenance using machine learning can forecast when fire safety
        equipment (sprinklers, alarms, extinguishers) will need maintenance, reducing failures during emergencies.
        """)
        
        # Interactive demo
        st.subheader("Predictive Maintenance Demo")
        
        equipment_type = st.selectbox(
            "Select Equipment Type:",
            ["Sprinkler System", "Fire Alarm", "Fire Extinguisher", "Smoke Detector"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            installation_date = st.date_input("Installation Date", value=pd.to_datetime("2023-01-01").date())
            last_inspection = st.date_input("Last Inspection Date", value=pd.to_datetime("2024-01-01").date())
            usage_intensity = st.slider("Usage Intensity", 1, 10, 5, 
                                      help="1 = Low usage environment, 10 = High usage/stress environment")
        
        with col2:
            environment = st.selectbox("Environment", ["Office", "Industrial", "Residential", "Commercial"])
            failures = st.number_input("Previous Failures (if any)", 0, 10, 0)
            temperature = st.slider("Average Ambient Temperature (°C)", 0, 40, 22)
        
        # Generate synthetic data for visualization
        today = pd.Timestamp.now().date()
        days_since_install = (today - installation_date).days
        days_since_inspection = (today - last_inspection).days
        
        # Equipment type factors
        type_factors = {
            "Sprinkler System": {"base_lifetime": 3650, "inspection_interval": 365, "failure_rate": 0.005},
            "Fire Alarm": {"base_lifetime": 1825, "inspection_interval": 180, "failure_rate": 0.01},
            "Fire Extinguisher": {"base_lifetime": 1825, "inspection_interval": 365, "failure_rate": 0.008},
            "Smoke Detector": {"base_lifetime": 730, "inspection_interval": 180, "failure_rate": 0.02}
        }
        
        # Environment factors
        env_factors = {
            "Office": 1.0,
            "Residential": 1.2,
            "Commercial": 1.3,
            "Industrial": 1.5
        }
        
        # Calculate health score and prediction
        base_lifetime = type_factors[equipment_type]["base_lifetime"]
        inspection_interval = type_factors[equipment_type]["inspection_interval"]
        
        # Adjust for environment and usage
        adjusted_lifetime = base_lifetime / (env_factors[environment] * (usage_intensity / 5))
        
        # Effect of temperature (higher temp = shorter life)
        if temperature > 25:
            adjusted_lifetime *= (1 - (temperature - 25) * 0.02)
        
        # Effect of failures
        if failures > 0:
            adjusted_lifetime *= (1 - failures * 0.1)
        
        # Current health calculation
        current_health = 100 - (days_since_install / adjusted_lifetime * 100)
        current_health = max(0, min(100, current_health))
        
        # Days to next maintenance
        days_to_maintenance = max(0, inspection_interval - days_since_inspection)
        
        # Probability of failure
        failure_rate = type_factors[equipment_type]["failure_rate"]
        adjusted_failure_rate = failure_rate * (env_factors[environment] * (usage_intensity / 5))
        
        # Increase with temperature and previous failures
        if temperature > 25:
            adjusted_failure_rate *= (1 + (temperature - 25) * 0.05)
        
        adjusted_failure_rate *= (1 + failures * 0.2)
        
        # Probability increases as health decreases
        failure_probability = adjusted_failure_rate * (1 + (100 - current_health) / 50)
        failure_probability = min(0.99, failure_probability)
        
        # Display health metrics with unified theme
        st.subheader("Equipment Health Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Current health gauge with unified styling
            health_color = FIRESAFETY_COLORS['secondary'] if current_health > 70 else FIRESAFETY_COLORS['warning'] if current_health > 30 else FIRESAFETY_COLORS['error']
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_health,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Current Health"},
                number = {'font': {'color': health_color, 'size': 36}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': FIRESAFETY_COLORS['text']},
                    'bar': {'color': health_color},
                    'bgcolor': FIRESAFETY_COLORS['background'],
                    'borderwidth': 2,
                    'bordercolor': FIRESAFETY_COLORS['light_text'],
                    'steps': [
                        {'range': [0, 30], 'color': "#FFEBEE"},  # Light red
                        {'range': [30, 70], 'color': "#FFF8E1"},  # Light yellow
                        {'range': [70, 100], 'color': "#E8F5E9"}  # Light green
                    ],
                    'threshold': {
                        'line': {'color': FIRESAFETY_COLORS['text'], 'width': 2},
                        'thickness': 0.75,
                        'value': current_health
                    }
                }
            ))
            
            # Apply unified theme
            fig = apply_firesafety_theme_to_plotly(
                fig,
                height=250
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Days to maintenance with unified styling
            maintenance_color = FIRESAFETY_COLORS['secondary'] if days_to_maintenance > 60 else FIRESAFETY_COLORS['warning'] if days_to_maintenance > 30 else FIRESAFETY_COLORS['error']
            fig = go.Figure(go.Indicator(
                mode = "number+delta",
                value = days_to_maintenance,
                title = {'text': "Days to Next Maintenance"},
                delta = {'reference': inspection_interval, 'relative': True, 'valueformat': '.1%'},
                number = {'suffix': " days", 'font': {'color': maintenance_color, 'size': 36}}
            ))
            
            # Apply unified theme
            fig = apply_firesafety_theme_to_plotly(
                fig,
                height=250
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Failure probability with unified styling
            probability_color = FIRESAFETY_COLORS['secondary'] if failure_probability < 0.05 else FIRESAFETY_COLORS['warning'] if failure_probability < 0.15 else FIRESAFETY_COLORS['error']
            fig = go.Figure(go.Indicator(
                mode = "number+gauge",
                value = failure_probability * 100,
                title = {'text': "Failure Risk (%)"},
                number = {'suffix': "%", 'font': {'color': probability_color, 'size': 36}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': FIRESAFETY_COLORS['text']},
                    'bar': {'color': probability_color},
                    'bgcolor': FIRESAFETY_COLORS['background'],
                    'borderwidth': 2,
                    'bordercolor': FIRESAFETY_COLORS['light_text'],
                    'steps': [
                        {'range': [0, 5], 'color': "#E8F5E9"},  # Light green
                        {'range': [5, 15], 'color': "#FFF8E1"},  # Light yellow
                        {'range': [15, 100], 'color': "#FFEBEE"}  # Light red
                    ],
                    'threshold': {
                        'line': {'color': FIRESAFETY_COLORS['text'], 'width': 2},
                        'thickness': 0.75,
                        'value': failure_probability * 100
                    }
                }
            ))
            
            # Apply unified theme
            fig = apply_firesafety_theme_to_plotly(
                fig,
                height=250
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance recommendations
        st.subheader("Maintenance Recommendations")
        
        if days_to_maintenance <= 0:
            st.error("⚠️ IMMEDIATE MAINTENANCE REQUIRED: Regular inspection interval exceeded")
        elif current_health < 30:
            st.error("⚠️ IMMEDIATE MAINTENANCE REQUIRED: Equipment health critical")
        elif current_health < 50:
            st.warning("⚠️ SCHEDULE MAINTENANCE SOON: Equipment performance degrading")
        elif days_to_maintenance < 30:
            st.warning("⚠️ MAINTENANCE COMING UP: Schedule inspection within a month")
        else:
            st.success("✓ Equipment in good condition: No immediate maintenance needed")
            
        # Show health trend
        st.subheader("Projected Health Trend")
        
        # Generate projected health data
        days = 365
        date_range = pd.date_range(start=pd.Timestamp.now(), periods=days)
        health_values = []
        
        for i in range(days):
            projected_days_since_install = days_since_install + i
            # Add expected maintenance effects
            maintenance_effect = 0
            if days_to_maintenance > 0:
                if i > days_to_maintenance and (i - days_to_maintenance) % inspection_interval == 0:
                    maintenance_effect = 20  # Maintenance improves health
            
            projected_health = 100 - (projected_days_since_install / adjusted_lifetime * 100) + maintenance_effect
            projected_health = max(0, min(100, projected_health))
            health_values.append(projected_health)
        
        health_df = pd.DataFrame({
            'Date': date_range,
            'Health': health_values
        })
        
        # Add maintenance markers
        maintenance_dates = []
        maintenance_health = []
        
        if days_to_maintenance > 0:
            current = days_to_maintenance
            while current < days:
                maintenance_dates.append(date_range[current])
                maintenance_health.append(health_values[current])
                current += inspection_interval
        
        # Plot projected health with our unified style
        fig = px.line(
            health_df, 
            x='Date', 
            y='Health', 
            color_discrete_sequence=[FIRESAFETY_COLORS['tertiary']]
        )
        
        # Add threshold lines with consistent colors
        fig.add_shape(
            type="line",
            y0=50, y1=50,
            x0=date_range[0], x1=date_range[-1],
            line=dict(color=FIRESAFETY_COLORS['warning'], width=2, dash="dash")
        )
        
        fig.add_shape(
            type="line",
            y0=30, y1=30,
            x0=date_range[0], x1=date_range[-1],
            line=dict(color=FIRESAFETY_COLORS['error'], width=2, dash="dash")
        )
        
        # Add maintenance markers with consistent styling
        if maintenance_dates:
            fig.add_trace(go.Scatter(
                x=maintenance_dates,
                y=maintenance_health,
                mode='markers',
                marker=dict(
                    size=12, 
                    color=FIRESAFETY_COLORS['secondary'], 
                    symbol='star',
                    line=dict(width=1, color='white')
                ),
                name='Scheduled Maintenance',
                hovertemplate='Date: %{x|%b %d, %Y}<br>Health Score: %{y:.1f}<extra>Maintenance</extra>'
            ))
        
        # Apply the unified theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title='Projected Equipment Health Over Time',
            height=450,
            legend_title="Events"
        )
        
        # Add annotations for thresholds with consistent styling
        fig.add_annotation(
            x=date_range[-1], y=50,
            text="Warning Threshold",
            showarrow=False,
            xshift=10,
            font=dict(color=FIRESAFETY_COLORS['warning'], size=12, family="Arial, sans-serif")
        )
        
        fig.add_annotation(
            x=date_range[-1], y=30,
            text="Critical Threshold",
            showarrow=False,
            xshift=10,
            font=dict(color=FIRESAFETY_COLORS['error'], size=12, family="Arial, sans-serif")
        )
        
        # Additional customizations
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Equipment Health Score',
            yaxis=dict(range=[0, 105]),  # Add a little padding above 100%
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        

        st.markdown('</div>', unsafe_allow_html=True)
    
    # Example 5: Fire Spread Simulation
    with example_tabs[4]:
        st.markdown('<div class="example-container example-5">', unsafe_allow_html=True)
        st.header("Fire Spread Prediction & Simulation")
        
        st.write("""
        This example demonstrates how machine learning combined with physics-based models can predict
        the spread of fires in different environments, helping in both planning and active firefighting.
        """)
        
        # Interactive simulation parameters
        st.subheader("Fire Spread Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            building_material = st.selectbox(
                "Building Material:",
                ["Wood Frame", "Concrete", "Steel", "Brick", "Mixed Materials"]
            )
            wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
            humidity = st.slider("Relative Humidity (%)", 0, 100, 40)
        
        with col2:
            ignition_point = st.selectbox(
                "Fire Ignition Point:",
                ["Kitchen", "Living Room", "Bedroom", "Garage", "Basement"]
            )
            temperature = st.slider("Ambient Temperature (°C)", 0, 40, 25)
            sprinklers = st.checkbox("Active Sprinkler System", True)
        
        # Material properties (simplified for demonstration)
        material_properties = {
            "Wood Frame": {"flammability": 0.8, "heat_resistance": 0.3},
            "Concrete": {"flammability": 0.1, "heat_resistance": 0.9},
            "Steel": {"flammability": 0.2, "heat_resistance": 0.7},
            "Brick": {"flammability": 0.3, "heat_resistance": 0.8},
            "Mixed Materials": {"flammability": 0.5, "heat_resistance": 0.5}
        }
        
        # Ignition point factors
        ignition_factors = {
            "Kitchen": {"fuel_load": 0.7, "containment": 0.4},
            "Living Room": {"fuel_load": 0.8, "containment": 0.5},
            "Bedroom": {"fuel_load": 0.6, "containment": 0.6},
            "Garage": {"fuel_load": 0.9, "containment": 0.3},
            "Basement": {"fuel_load": 0.7, "containment": 0.2}
        }
        
        # Environmental factors
        env_factor = 1.0
        if humidity < 30:
            env_factor *= (1 + (30 - humidity) * 0.02)
        if temperature > 30:
            env_factor *= (1 + (temperature - 30) * 0.03)
        if wind_speed > 20:
            env_factor *= (1 + (wind_speed - 20) * 0.04)
            
        # Sprinkler effect
        sprinkler_factor = 0.4 if sprinklers else 1.0
        
        # Calculate base spread rate
        flammability = material_properties[building_material]["flammability"]
        heat_resistance = material_properties[building_material]["heat_resistance"]
        fuel_load = ignition_factors[ignition_point]["fuel_load"]
        containment = ignition_factors[ignition_point]["containment"]
        
        spread_rate = (flammability * fuel_load * env_factor * sprinkler_factor) / (heat_resistance * containment)
        spread_rate = max(0.1, min(1.0, spread_rate))
        
        # Time to critical thresholds
        time_to_room = 5 / spread_rate
        time_to_floor = 15 / spread_rate
        time_to_building = 30 / spread_rate
        
        # Create time simulation for visualization
        st.subheader("Fire Spread Timeline Simulation")
        
        max_time = int(time_to_building * 1.2)
        time_points = list(range(0, max_time + 1, 1))
        
        # Function to calculate fire spread area
        def fire_area(t, max_area=100, rate=spread_rate):
            if t == 0:
                return 0
            logistic_factor = 10 / (time_to_building * 0.6)
            return max_area / (1 + np.exp(-logistic_factor * (t - time_to_building * 0.4)))
        
        # Create DataFrame for visualization
        spread_df = pd.DataFrame({
            'Time (minutes)': time_points,
            'Fire Area (%)': [fire_area(t) for t in time_points]
        })
        
        # Plot fire spread over time with unified styling
        fig = px.line(
            spread_df, 
            x='Time (minutes)', 
            y='Fire Area (%)',
            color_discrete_sequence=[FIRESAFETY_COLORS['primary']]
        )
        
        # Add threshold markers with consistent colors
        fig.add_vline(x=time_to_room, line_width=2, line_dash="dash", line_color=FIRESAFETY_COLORS['warning'])
        fig.add_vline(x=time_to_floor, line_width=2, line_dash="dash", line_color=FIRESAFETY_SEQUENTIAL[6])
        fig.add_vline(x=time_to_building, line_width=2, line_dash="dash", line_color=FIRESAFETY_COLORS['error'])
        
        # Apply the unified theme
        fig = apply_firesafety_theme_to_plotly(
            fig,
            title="Predicted Fire Spread Over Time",
            height=450
        )
        
        # Add annotations with consistent styling
        fig.add_annotation(
            x=time_to_room, y=20,
            text="Room Engulfed",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=FIRESAFETY_COLORS['warning'],
            font=dict(family="Arial, sans-serif", size=12, color=FIRESAFETY_COLORS['warning'])
        )
        
        fig.add_annotation(
            x=time_to_floor, y=40,
            text="Floor Engulfed",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=FIRESAFETY_SEQUENTIAL[6],
            font=dict(family="Arial, sans-serif", size=12, color=FIRESAFETY_SEQUENTIAL[6])
        )
        
        fig.add_annotation(
            x=time_to_building, y=70,
            text="Building Compromised",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=FIRESAFETY_COLORS['error'],
            font=dict(family="Arial, sans-serif", size=12, color=FIRESAFETY_COLORS['error'])
        )
        
        # Customize hover template
        fig.update_traces(
            hovertemplate='Time: %{x} minutes<br>Fire Area: %{y:.1f}%<extra></extra>'
        )
        
        # Additional customizations
        fig.update_layout(
            xaxis_title='Time (minutes)',
            yaxis_title='Fire Spread (%)',
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display critical times
        st.subheader("Critical Timepoints")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Time to Room Engulfed",
                f"{time_to_room:.1f} min",
                delta="-2.0 min" if sprinklers else None,
                delta_color="inverse"
            )
            
        with col2:
            st.metric(
                "Time to Floor Engulfed",
                f"{time_to_floor:.1f} min",
                delta="-5.0 min" if sprinklers else None,
                delta_color="inverse"
            )
            
        with col3:
            st.metric(
                "Time to Building Compromised",
                f"{time_to_building:.1f} min",
                delta="-10.0 min" if sprinklers else None,
                delta_color="inverse"
            )
        
        # How it works
        st.subheader("How Fire Spread Prediction Works")
        
        st.write("""
        The fire spread prediction model combines:
        
        1. **Physics-Based Fire Dynamics**: Using computational fluid dynamics to model heat transfer, combustion, and smoke movement
        
        2. **Material Properties Database**: Including flammability, thermal conductivity, and heat capacity for various building materials
        
        3. **ML-Enhanced Parameter Estimation**: Machine learning that's been trained on thousands of real fire incidents to predict parameters 
           that are difficult to measure directly
        
        4. **Environmental Factors**: Including temperature, humidity, wind, and building ventilation
        
        5. **Mitigation Systems**: Accounting for sprinklers, fire barriers, and other suppression systems
        
        The model can predict not just the speed of fire spread, but also:
        - Temperature distribution over time
        - Smoke propagation paths
        - Structural integrity timelines
        - Safe evacuation window
        """)
        

        
        # Practical applications
        st.subheader("Practical Applications")
        
        st.write("""
        This technology supports:
        
        - **Building Design Optimization**: Testing different layouts and materials to maximize fire safety
        - **Emergency Response Planning**: Providing firefighters with spread predictions during active incidents
        - **Insurance Risk Assessment**: More accurate evaluation of building fire risks
        - **Real-time Decision Support**: When connected to IoT sensors, the system can update predictions as an incident unfolds
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Return to learning dashboard button
    st.button("Return to Learning Dashboard", on_click=lambda: set_page('dashboard'), use_container_width=True)

# Helper functions for page navigation
def set_page(page):
    st.session_state.page = page
    
def set_page_with_module(page, module_index):
    st.session_state.page = page
    st.session_state.current_module = module_index
    
def logout_user():
    """Logout the user by resetting session state"""
    # Save any unsaved progress if user is authenticated
    if st.session_state.authenticated and not st.session_state.is_guest:
        # Update tracked time for the last module if there is one
        if hasattr(st.session_state, 'current_tracking'):
            module_id = st.session_state.current_module
            elapsed_time = (time.time() - st.session_state.current_tracking['start_time']) / 60  # Convert to minutes
            # Only update if significant time has passed (more than 1 minute)
            if elapsed_time > 1:
                update_user_progress(
                    st.session_state.user_id,
                    module_id,
                    st.session_state.module_progress.get(module_id, 0),
                    time_spent=int(elapsed_time)
                )
    
    # Clear session variables
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.user_name = None
    st.session_state.page = 'login'
    st.session_state.current_module = 0
    st.session_state.module_progress = {}
    st.session_state.is_guest = False
    
    # Clear any tracking data
    if hasattr(st.session_state, 'current_tracking'):
        del st.session_state.current_tracking
    if hasattr(st.session_state, 'module_progress_details'):
        del st.session_state.module_progress_details

# Main app flow logic
def main():
    # Load custom CSS for styling
    load_css()
    
    if not st.session_state.authenticated:
        # Remove tabs navigation as requested
        if 'auth_page' not in st.session_state:
            st.session_state.auth_page = 'login'
            
        if st.session_state.auth_page == 'login':
            show_login_page()
        elif st.session_state.auth_page == 'register':
            show_registration_page()
        elif st.session_state.auth_page == 'reset':
            show_reset_password_page()
    else:
        # Load user progress if just authenticated
        if not st.session_state.module_progress:
            progress = get_user_progress(st.session_state.user_id)
            st.session_state.module_progress = progress
        
        show_main_app()

if __name__ == "__main__":
    main()