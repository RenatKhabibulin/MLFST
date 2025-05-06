import streamlit as st
import hashlib
import uuid
import re
from database import get_db_connection
from psycopg2.extras import RealDictCursor

def hash_password(password):
    """Hash a password for storing."""
    salt = uuid.uuid4().hex
    return hashlib.sha256(salt.encode() + password.encode()).hexdigest() + ':' + salt

def check_password_hash(hashed_password, user_password):
    """Check a stored password against one provided by user"""
    password, salt = hashed_password.split(':')
    return password == hashlib.sha256(salt.encode() + user_password.encode()).hexdigest()

def is_valid_email(email):
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

def authenticate_user(email, password):
    """Authenticate a user"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get user with the provided email
    cursor.execute("SELECT id, name, password FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    
    conn.close()
    
    if user and check_password_hash(user['password'], password):
        return {'id': user['id'], 'name': user['name']}
    
    return None

def register_user(name, email, password):
    """Register a new user"""
    # Check if email already exists
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    if cursor.fetchone():
        conn.close()
        return False, "Email already registered"
    
    # Hash password and store user
    hashed_password = hash_password(password)
    try:
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_password)
        )
        conn.commit()
        conn.close()
        return True, "Registration successful"
    except Exception as e:
        conn.close()
        return False, f"Registration failed: {str(e)}"

def reset_password(email, new_password):
    """Reset user password"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Check if email exists
    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    if not cursor.fetchone():
        conn.close()
        return False, "Email not found"
    
    # Update password
    hashed_password = hash_password(new_password)
    try:
        cursor.execute(
            "UPDATE users SET password = %s WHERE email = %s",
            (hashed_password, email)
        )
        conn.commit()
        conn.close()
        return True, "Password reset successful"
    except Exception as e:
        conn.close()
        return False, f"Password reset failed: {str(e)}"

def show_login_page():
    """Display login form with improved design"""
    
    # Add custom CSS for an attractive login page
    st.markdown("""
    <style>
    .login-page {
        display: block;
        margin: 0 auto;
        padding: 20px;
        max-width: 100%;
    }
    
    .app-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .app-title {
        font-size: 3.5em;
        font-weight: 700;
        color: #FF5722;
        margin-bottom: 5px;
    }
    
    .app-subtitle {
        font-size: 1.5em;
        color: #555;
    }
    
    /* Platform overview section */
    .platform-overview {
        width: 100%;
        background-color: white;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 30px;
        text-align: center;
        display: block; /* Ensure block display */
    }
    
    .section-title {
        color: #333;
        font-size: 1.8em;
        margin-bottom: 15px;
        font-weight: 600;
    }
    
    .platform-description {
        color: #555;
        font-size: 1.1em;
        margin-bottom: 25px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Graphics mosaic - completely restructured for maximum compatibility */
    .graphics-mosaic {
        display: block;
        width: 100%;
        margin: 30px auto;
        max-width: 900px;
    }
    
    .mosaic-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        width: 100%;
        margin-bottom: 15px;
        gap: 20px;
    }
    
    .mosaic-item {
        flex: 1 1 200px;
        min-width: 200px;
        max-width: calc(50% - 20px);
        margin: 0 0 20px 0;
        background-color: #ffffff;
        border-radius: 8px;
        overflow: hidden;
        padding: 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    @media (max-width: 640px) {
        .mosaic-item {
            flex-basis: 100%;
            max-width: 100%;
        }
    }
    
    .mosaic-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    
    .mosaic-chart {
        padding: 0;
        overflow: hidden;
        width: 100%;
        height: 100%;
    }
    
    /* CSS debugging styles */
    .debug-outline {
        border: 1px solid red !important;
    }
    
    /* Guest access button styling */
    .guest-access-container {
        width: 100%;
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }
    
    /* Make the Guest button larger and centered */
    .guest-access-container {
        text-align: center;
    }
    
    .guest-access-container button {
        background-color: #FF5722 !important;
        color: white !important;
        font-size: 1.45em !important; /* 10% larger than before */
        padding: 18px 55px !important; /* increased padding */
        border: none !important;
        border-radius: 30px !important;
        box-shadow: 0 5px 10px rgba(255, 87, 34, 0.4) !important;
        transition: all 0.3s ease !important;
        width: auto !important;
        min-width: 275px !important; /* 10% wider than before */
        margin: 0 auto !important;
        display: block !important;
    }
    
    .guest-access-container button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(255, 87, 34, 0.4) !important;
    }
    
    /* Login section */
    .login-section {
        width: 100%;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .content-area {
        display: flex;
        width: 100%;
        max-width: 1200px;
        gap: 30px;
    }
    
    .login-container {
        background-color: white;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-top: 5px solid #FF5722;
        flex: 1;
    }
    
    .features-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border-top: 5px solid #4CAF50;
        flex: 1;
    }
    
    .feature-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 15px;
    }
    
    .feature-icon {
        font-size: 1.5em;
        margin-right: 10px;
        color: #4CAF50;
    }
    
    .form-title {
        color: #333;
        font-size: 1.8em;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #777;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main login page layout with new design
    st.markdown('<div class="login-page">', unsafe_allow_html=True)
    
    # App header with platform title
    st.markdown('''
    <div class="app-header">
        <h1 class="app-title">ML FireSafetyTutor</h1>
        <h2 class="app-subtitle">— Educational Platform</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    # Platform overview with simplified visuals
    st.markdown('''
    <div class="platform-overview">
        <h2 class="section-title">Interactive Learning Platform for AI in Fire Safety</h2>
        <p class="platform-description">
            Explore machine learning concepts and applications in fire safety through 
            interactive modules, visualizations, and practical examples.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Visual cards with charts above them
    st.write("## Key Machine Learning Methods")
    
    # Create sample data for visualization
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from visualization import apply_firesafety_theme_to_plotly, FIRESAFETY_COLORS, FIRESAFETY_SEQUENTIAL, FIRESAFETY_CATEGORICAL
    
    # Column 1 - Fire Incident Chart
    col1, col2 = st.columns(2)
    with col1:
        # Create a sample fire incidents bar chart
        causes = ['Electrical', 'Cooking', 'Heating', 'Smoking', 'Arson']
        counts = [45, 30, 25, 15, 10]
        df_causes = pd.DataFrame({'Cause': causes, 'Count': counts})
        
        fig1 = px.bar(
            df_causes, 
            x='Cause', 
            y='Count',
            color='Count',
            color_continuous_scale=FIRESAFETY_SEQUENTIAL
        )
        
        # Apply our fire safety theme
        fig1 = apply_firesafety_theme_to_plotly(
            fig1, 
            title='Fire Incidents by Cause',
            height=300,
        )
        
        fig1.update_layout(
            xaxis_title='',
            yaxis_title='',
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        
        # Information card
        st.info("**🔥 Fire Incident Analytics**\n\nUse statistical methods to identify patterns and trends in fire incident data")
    
    # Column 2 - Classification Chart
    with col2:
        # Create a sample classification scatter plot
        np.random.seed(42)
        n = 50
        df_class = pd.DataFrame({
            'Risk Factor 1': np.random.normal(5, 1.5, n),
            'Risk Factor 2': np.random.normal(7, 2, n),
            'Risk Level': np.random.choice(['High', 'Medium', 'Low'], n, 
                                        p=[0.3, 0.5, 0.2])
        })
        
        fig2 = px.scatter(
            df_class, 
            x='Risk Factor 1', 
            y='Risk Factor 2',
            color='Risk Level',
            color_discrete_sequence=FIRESAFETY_CATEGORICAL,
            opacity=0.8
        )
        
        # Apply our fire safety theme
        fig2 = apply_firesafety_theme_to_plotly(
            fig2, 
            title='Building Risk Classification',
            height=300,
            legend_title='Risk Level'
        )
        
        fig2.update_layout(
            xaxis_title='',
            yaxis_title='',
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Information card
        st.success("**🏢 Building Classification**\n\nClassify buildings by fire risk level using supervised learning techniques")
    
    # Column 3 - Regression Chart
    col3, col4 = st.columns(2)
    with col3:
        # Create a sample regression line plot
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y = 2 * x + 1 + np.random.normal(0, 1.5, 30)
        df_reg = pd.DataFrame({'Time': x, 'Spread': y})
        
        fig3 = px.scatter(
            df_reg, 
            x='Time', 
            y='Spread',
            opacity=0.7,
            color_discrete_sequence=[FIRESAFETY_COLORS['secondary']]
        )
        
        # Add regression line
        fig3.add_trace(
            go.Scatter(
                x=x,
                y=2*x + 1,
                mode='lines',
                name='Trend',
                line=dict(color=FIRESAFETY_COLORS['primary'], width=2)
            )
        )
        
        # Apply our fire safety theme
        fig3 = apply_firesafety_theme_to_plotly(
            fig3, 
            title='Fire Spread Prediction',
            height=300
        )
        
        fig3.update_layout(
            xaxis_title='',
            yaxis_title='',
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Information card
        st.info("**📉 Predictive Modeling**\n\nApply regression analysis to predict fire spread and damage potential")
    
    # Column 4 - Neural Network Visualization
    with col4:
        # Create a simplified neural network visualization
        fig4 = go.Figure()
        
        # Layers coordinates
        layer_x = [0, 1, 2]
        layers = [
            [1, 2, 3, 4],  # Input layer
            [1, 2, 3],     # Hidden layer
            [1, 2]         # Output layer
        ]
        
        # Draw nodes
        for i, layer in enumerate(layers):
            for j, node in enumerate(layer):
                y_pos = (len(layer) - 1) / 2 - j
                
                # Add nodes as circles
                fig4.add_trace(
                    go.Scatter(
                        x=[layer_x[i]],
                        y=[y_pos],
                        mode='markers',
                        marker=dict(
                            size=15,
                            color=FIRESAFETY_COLORS['tertiary' if i == 1 else 'secondary'],
                            line=dict(width=1, color='white')
                        ),
                        showlegend=False
                    )
                )
        
        # Draw connections between nodes
        for i in range(len(layers) - 1):
            for j in range(len(layers[i])):
                for k in range(len(layers[i + 1])):
                    y1 = (len(layers[i]) - 1) / 2 - j
                    y2 = (len(layers[i + 1]) - 1) / 2 - k
                    
                    fig4.add_trace(
                        go.Scatter(
                            x=[layer_x[i], layer_x[i + 1]],
                            y=[y1, y2],
                            mode='lines',
                            line=dict(color=FIRESAFETY_COLORS['light_text'], width=1),
                            showlegend=False
                        )
                    )
        
        # Apply our fire safety theme
        fig4 = apply_firesafety_theme_to_plotly(
            fig4, 
            title='Neural Network Architecture',
            height=300
        )
        
        fig4.update_layout(
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-0.5, 2.5]
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                range=[-2, 2]
            ),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Information card
        st.success("**🔄 Deep Learning**\n\nLeverage neural networks for complex image-based fire detection systems")
    
    # Large Guest Access button in the center - Stylized version
    st.markdown("""
    <style>
    .guest-access-container {
        width: 100%;
        display: flex;
        justify-content: center;
        margin: 40px 0;
    }
    
    /* CSS for centering and styling the button */
    .centered-guest-button {
        display: block !important;
        margin: 0 auto !important;
        max-width: 400px !important;
        background-color: #FF5722 !important;
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
        padding: 0.8rem 2rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 10px rgba(255, 87, 34, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    /* Hover effect */
    .centered-guest-button:hover {
        background-color: #F4511E !important;
        box-shadow: 0 6px 15px rgba(255, 87, 34, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add custom CSS for the button first
    st.markdown("""
    <style>
    /* Override Streamlit's default button styling */
    .stButton > button {
        background-color: #FF5722 !important;
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        padding: 0.8rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(255, 87, 34, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        max-width: 400px !important;
        display: block !important;
        margin: 0 auto !important;
    }
    
    .stButton > button:hover {
        background-color: #F4511E !important;
        box-shadow: 0 6px 15px rgba(255, 87, 34, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Center the container */
    .centered-button-container {
        display: flex !important;
        justify-content: center !important;
        margin: 40px auto !important;
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create a centered column for the button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # This is the simplest approach - just use Streamlit's native button but with custom CSS
        if st.button("Explore as Guest", key="guest_button", use_container_width=True):
            # Update all relevant session state variables
            st.session_state.authenticated = True
            st.session_state.user_id = None
            st.session_state.user_name = "Guest"
            st.session_state.page = 'module'  # Skip dashboard, go directly to first module
            st.session_state.current_module = 0  # Set to first module (Introduction to ML in Fire Safety)
            st.session_state.is_guest = True
            st.success("Welcome, Guest! Redirecting to Module 1...")
            
            # Use Streamlit's native rerun function instead of JavaScript - more reliable
            st.rerun()
            return True
    
    # Add spacing
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Login section
    st.markdown('<div class="login-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">User Access</h2>', unsafe_allow_html=True)
    
    # Two columns for Login and Features
    col1, col2 = st.columns(2)
    
    # Login container
    with col1:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="form-title">Login</h3>', unsafe_allow_html=True)
        
        email = st.text_input("Email", key="login_email", 
                             placeholder="your-email@example.com")
        password = st.text_input("Password", type="password", key="login_password",
                                placeholder="Your password")
        
        if st.button("Login", key="login_button", use_container_width=True):
            if not email or not password:
                st.error("Please enter email and password")
                return False
            
            user = authenticate_user(email, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user['id']
                st.session_state.user_name = user['name']
                st.session_state.page = 'dashboard'
                st.session_state.is_guest = False
                st.success("Login successful!")
                return True
            else:
                st.error("Invalid email or password")
                return False
        
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        # Register and reset buttons
        reg_col1, reg_col2 = st.columns(2)
        with reg_col1:
            if st.button("Create Account", key="register_button", use_container_width=True):
                st.session_state.auth_page = 'register'
                st.rerun()
        
        with reg_col2:
            if st.button("Reset Password", key="reset_button", use_container_width=True):
                st.session_state.auth_page = 'reset'
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Features container
    with col2:
        st.markdown('<div class="features-container">', unsafe_allow_html=True)
        st.markdown('<h3 class="form-title">Platform Features</h3>', unsafe_allow_html=True)
        
        # Feature items
        st.markdown("""
        <div class="feature-item">
            <div class="feature-icon">🔍</div>
            <div>
                <strong>Interactive Learning Modules</strong>
                <p>Explore ML methods in fire safety</p>
            </div>
        </div>
        
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div>
                <strong>Data Visualization</strong>
                <p>Interactive charts and diagrams</p>
            </div>
        </div>
        
        <div class="feature-item">
            <div class="feature-icon">🧪</div>
            <div>
                <strong>Practical Examples</strong>
                <p>Real-world ML applications</p>
            </div>
        </div>
        
        <div class="feature-item">
            <div class="feature-icon">📈</div>
            <div>
                <strong>Progress Tracking</strong>
                <p>Monitor your learning journey</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">© 2025 ML FireSafetyTutor — Educational Platform</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    return False

def show_registration_page():
    """Display registration form with improved design"""
    
    # Add custom CSS for an attractive registration page
    st.markdown("""
    <style>
    .registration-page {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
    }
    
    .app-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .app-title {
        font-size: 3.5em;
        font-weight: 700;
        color: #FF5722;
        margin-bottom: 5px;
    }
    
    .app-subtitle {
        font-size: 1.5em;
        color: #555;
    }
    
    .register-container {
        background-color: white;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-top: 5px solid #4B7BFF;
        max-width: 600px;
        width: 100%;
    }
    
    .form-title {
        color: #333;
        font-size: 1.8em;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    .field-info {
        font-size: 0.85em;
        color: #6c757d;
        margin-top: -15px;
        margin-bottom: 15px;
    }
    
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #777;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main registration page layout
    st.markdown('<div class="registration-page">', unsafe_allow_html=True)
    
    # App header with platform title
    st.markdown('''
    <div class="app-header">
        <h1 class="app-title">ML FireSafetyTutor</h1>
        <h2 class="app-subtitle">— Educational Platform</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    # Registration container
    st.markdown('<div class="register-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="form-title">Registration</h2>', unsafe_allow_html=True)
    
    name = st.text_input("Full Name", key="reg_name", 
                       placeholder="John Smith")
    
    email = st.text_input("Email", key="reg_email", 
                        placeholder="your-email@example.com")
    st.markdown('<div class="field-info">Will be used for login and notifications</div>', 
              unsafe_allow_html=True)
    
    password = st.text_input("Password", type="password", key="reg_password")
    st.markdown('<div class="field-info">Minimum 6 characters required</div>', 
              unsafe_allow_html=True)
    
    confirm_password = st.text_input("Confirm Password", type="password", 
                                   key="reg_confirm")
    
    # Registration and back buttons
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Registration button with full width
        if st.button("Create Account", key="reg_button", use_container_width=True):
            # Validate inputs
            if not name or not email or not password:
                st.error("Please fill in all required fields")
                return
            
            if password != confirm_password:
                st.error("Passwords do not match")
                return
                
            if not is_valid_email(email):
                st.error("Please enter a valid email address")
                return
                
            if len(password) < 6:
                st.error("Password must be at least 6 characters long")
                return
            
            # Register user
            success, message = register_user(name, email, password)
            if success:
                st.success("Registration successful")
                st.session_state.auth_page = 'login'
                st.rerun()
            else:
                st.error(f"Registration error: {message}")
    
    with col2:
        if st.button("Back", key="back_button", use_container_width=True):
            st.session_state.auth_page = 'login'
            st.rerun()
    
    # Additional info below the form
    st.markdown("""
    <div style="text-align: center; margin-top: 25px; color: #6c757d;">
    After registration, you will be able to track your progress and take tests.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">© 2025 ML FireSafetyTutor — Educational Platform</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_reset_password_page():
    """Display password reset form with improved design"""
    
    # Add custom CSS for an attractive reset page
    st.markdown("""
    <style>
    .reset-page {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
    }
    
    .app-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .app-title {
        font-size: 3.5em;
        font-weight: 700;
        color: #FF5722;
        margin-bottom: 5px;
    }
    
    .app-subtitle {
        font-size: 1.5em;
        color: #555;
    }
    
    .reset-container {
        background-color: white;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-top: 5px solid #FFC107;
        max-width: 600px;
        width: 100%;
    }
    
    .form-title {
        color: #333;
        font-size: 1.8em;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    .field-info {
        font-size: 0.85em;
        color: #6c757d;
        margin-top: -15px;
        margin-bottom: 15px;
    }
    
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #777;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main reset page layout
    st.markdown('<div class="reset-page">', unsafe_allow_html=True)
    
    # App header with platform title
    st.markdown('''
    <div class="app-header">
        <h1 class="app-title">ML FireSafetyTutor</h1>
        <h2 class="app-subtitle">— Educational Platform</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    # Reset password container
    st.markdown('<div class="reset-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="form-title">Reset Password</h2>', unsafe_allow_html=True)
    
    email = st.text_input("Email", key="reset_email", 
                         placeholder="Enter your registered email")
    
    new_password = st.text_input("New Password", type="password", 
                               key="reset_password")
    st.markdown('<div class="field-info">Minimum 6 characters required</div>', 
              unsafe_allow_html=True)
    
    confirm_password = st.text_input("Confirm Password", type="password", 
                                   key="reset_confirm")
    
    # Reset and back buttons
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Reset password button with full width
        if st.button("Reset Password", key="reset_btn", use_container_width=True):
            # Validate inputs
            if not email or not new_password:
                st.error("Please fill in all required fields")
                return
            
            if new_password != confirm_password:
                st.error("Passwords do not match")
                return
                
            if not is_valid_email(email):
                st.error("Please enter a valid email address")
                return
                
            if len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
                return
            
            # Reset password
            success, message = reset_password(email, new_password)
            if success:
                st.success("Password successfully changed")
                st.session_state.auth_page = 'login'
                st.rerun()
            else:
                st.error(f"Password reset error: {message}")
    
    with col2:
        if st.button("Back", key="back_btn", use_container_width=True):
            st.session_state.auth_page = 'login'
            st.rerun()
    
    # Additional info
    st.markdown("""
    <div style="text-align: center; margin-top: 25px; color: #6c757d;">
    After resetting your password, you will be able to log in with your new credentials.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown('<div class="footer">© 2025 ML FireSafetyTutor — Educational Platform</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
