import os
import sys

# Append the parent directory to the sys.path to ensure create_env is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_env import setup_environment

# Set up environment variables
setup_environment()

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

# Load environment variables
data_path = os.getenv('DATA_PATH', 'data/wine_quality.csv')
desired_test_score = float(os.getenv('DESIRED_TEST_SCORE', 0.85))
model_seed = int(os.getenv('MODEL_SEED', 42))
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Default_Experiment')
mlflow_model_name = os.getenv('MLFLOW_MODEL_NAME', 'Default_Model')
mlflow_model_stage = os.getenv('MLFLOW_MODEL_STAGE', 'None')

# Set MLflow tracking URI
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set up MLflow experiment
mlflow.set_experiment(experiment_name)

# Load data for validation
df = pd.read_csv(data_path)
y = df.pop('quality')
_, X_test, _, y_test = train_test_split(df, y, test_size=0.2, random_state=model_seed)

# Start an MLflow run to log the evaluation
with mlflow.start_run(run_name="Validation run"):
	mlflow.set_tag('purpose', 'validation')

	# Load model from MLflow Model Registry
	try:
		# Construct the model URI using the model name and stage
		mlflow_model_uri = f'models:/{mlflow_model_name}/{mlflow_model_stage}'
		model = mlflow.sklearn.load_model(mlflow_model_uri)
		
		# Evaluate the MLflow model
		test_score = model.score(X_test, y_test)
		mlflow.log_metric('Validation testing score', test_score)
		print(f"MLflow model test score: {test_score:.2f}")
		
	except Exception as e:
		mlflow.log_param('MLflow model loading error', str(e))
		print(f"An error occurred while loading the MLflow model: {e}")
		sys.exit(-1)
	
	# Verify if the model meets the desired accuracy threshold
	if test_score < desired_test_score:
		mlflow.log_param('Validation result', 'Failure')
		print(f"Test score {test_score:.2f} is below the desired threshold of {desired_test_score:.2f}%.")
		sys.exit(-1)

	mlflow.log_param('Validation result', 'Success')
	print("Model meets the required accuracy!")
	sys.exit(0)
