# Breast Cancer Detection — Multi-Agent System
# Each agent is a self-contained module with explicit dict-based I/O.

from .agent1_data_preprocessing import run as run_data_preprocessing
from .agent2_eda_analysis import run as run_eda_analysis
from .agent3_model_training import run as run_model_training
from .agent4_deployment import run as run_deployment
