"""
ML FireSafetyTutor - Interactive platform for learning ML in fire safety applications
"""

# Set package version
__version__ = "0.1.0"

# Import key modules to make them available through the package
import os
import sys

# Directory paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES_DIR = os.path.join(ROOT_DIR, "modules")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Add modules directory to path for importing
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

# Add parent directory to path for importing
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)