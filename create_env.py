import os
import configparser

def setup_environment():
	# Initialize ConfigParser
	config = configparser.ConfigParser()

	# Load the config.ini file
	config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
	
	# Check if the configuration file exists
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Configuration file not found at path: {config_path}")
	
	# Read configuration file and check for successful load
	config.read(config_path)

	# Add error checking for missing sections
	if 'PATHS' not in config:
		raise KeyError("Missing 'PATHS' section in configuration.")
	if 'MODEL' not in config:
		raise KeyError("Missing 'MODEL' section in configuration.")
	if 'MLFLOW' not in config:
		raise KeyError("Missing 'MLFLOW' section in configuration.")

	# Paths settings (only for data as models are now handled by MLflow)
	os.environ['DATA_PATH'] = config['PATHS'].get('DATA_PATH', '').strip().strip('"')
	if not os.environ['DATA_PATH']:
		raise KeyError("DATA_PATH is not defined in the 'PATHS' section.")

	# Model settings
	os.environ['DESIRED_TEST_SCORE'] = config['MODEL'].get('DESIRED_TEST_SCORE', '0.3').strip().strip('"')
	os.environ['MODEL_SEED'] = config['MODEL'].get('MODEL_SEED', '42').strip().strip('"')
	os.environ['RF_HYPERPARAM_SEARCH'] = config['MODEL'].get('RF_HYPERPARAM_SEARCH', '{}').strip().strip('"')

	# MLflow settings
	os.environ['MLFLOW_TRACKING_URI'] = config['MLFLOW'].get('MLFLOW_URI', '').strip().strip('"')
	os.environ['MLFLOW_EXPERIMENT_NAME'] = config['MLFLOW'].get('MLFLOW_EXPERIMENT_NAME', 'Default_Experiment').strip().strip('"')
	os.environ['MLFLOW_MODEL_NAME'] = config['MLFLOW'].get('MLFLOW_MODEL_NAME', 'Default_Model').strip().strip('"')
	os.environ['MLFLOW_MODEL_STAGE'] = config['MLFLOW'].get('MLFLOW_MODEL_STAGE', 'None').strip().strip('"')  # Default to None if not present
