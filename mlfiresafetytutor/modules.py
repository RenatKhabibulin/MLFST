"""
ML FireSafetyTutor modules interface
"""

# This module is a placeholder to ensure the package structure is complete
# The actual modules functionality is in the modules/ directory

import sys
import os

# Get the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES_DIR = os.path.join(ROOT_DIR, "modules")

# Add the modules directory to the Python path if it's not already there
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

# Import module functions for direct access
try:
    from modules.intro import show_module_content as show_intro
    from modules.data_prep import show_module_content as show_data_prep
    from modules.supervised import show_module_content as show_supervised
    from modules.unsupervised import show_module_content as show_unsupervised
    from modules.model_eval import show_module_content as show_model_eval
    from modules.deep_learning import show_module_content as show_deep_learning
    from modules.ai_agents import show_module_content as show_ai_agents
except ImportError:
    # For deployment fallback
    pass