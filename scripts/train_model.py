import os
import sys

# Append the parent directory to the sys.path to ensure create_env is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_env import setup_environment

# Set up environment variables
setup_environment()

import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
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

################################
########## DATA PREP ###########
################################

# Load the data
df = pd.read_csv(data_path)

# Split into train and test sections
y = df.pop('quality')
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=model_seed)

#################################
########## MODELING #############
#################################

# Start an MLflow run
with mlflow.start_run(run_name="Training run"):
	mlflow.set_tag('purpose', 'training')

	# Train the model
	model = RandomForestRegressor(max_depth=2, random_state=model_seed)
	model.fit(X_train, y_train)

	# Log parameters in MLflow
	mlflow.log_param('max_depth', 2)
	mlflow.log_param('random_state', model_seed)

	# Log the model in MLflow
	mlflow.sklearn.log_model(
		sk_model=model,
		artifact_path='model',
		registered_model_name=mlflow_model_name
	)
	
	print("Trained and logged a new model.")

	#################################
	########### METRICS #############
	#################################

	# Report training and test scores
	train_score = model.score(X_train, y_train)
	test_score = model.score(X_test, y_test)
	print(f"Training variance explained: {train_score:.2f}")
	print(f"Test variance explained: {test_score:.2f}")

	# Log metrics in MLflow
	mlflow.log_metric('Training score', train_score)
	mlflow.log_metric('Testing score', test_score)

print("Metrics logged in MLflow.")
