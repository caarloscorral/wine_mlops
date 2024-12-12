import os
import sys

# Append the parent directory to the sys.path to ensure create_env is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from create_env import setup_environment

# Set up environment variables
setup_environment()

import json
import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Load environment variables
data_path = os.getenv('DATA_PATH', 'data/wine_quality.csv')
desired_test_score = float(os.getenv('DESIRED_TEST_SCORE', 0.85))
rf_param_search = json.loads(os.getenv('RF_HYPERPARAM_SEARCH'))
model_seed = int(os.getenv('MODEL_SEED', 42))
mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Default_Experiment')
mlflow_model_name = os.getenv('MLFLOW_MODEL_NAME', 'Default_Model')
mlflow_model_stage = os.getenv('MLFLOW_MODEL_STAGE', 'None')

# Set MLflow tracking URI
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Set up MLflow experiment
mlflow.set_experiment(experiment_name)

# Load data
df = pd.read_csv(data_path)

# Split data
y = df.pop('quality')
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=model_seed)

# Start an MLflow run
with mlflow.start_run(run_name="Hyperparameter tuning"):
	mlflow.set_tag('purpose', 'hyperparameter tuning')

	# Define the model
	model = RandomForestRegressor(random_state=model_seed)

	# Log hyperparameter search space
	mlflow.log_param('rf_param_search', rf_param_search)

	# Perform grid search
	grid_search = GridSearchCV(
		estimator=model,
		param_grid=rf_param_search,
		cv=3,
		n_jobs=-1,
		verbose=0
	)
	grid_search.fit(X_train, y_train)

	# Get the best model
	best_model = grid_search.best_estimator_
	test_score = best_model.score(X_test, y_test)

	# Log best model parameters and test score
	mlflow.log_params(grid_search.best_params_)
	mlflow.log_metric('Hyperparameter tuning testing score', test_score)

	# Log the best model in MLflow
	mlflow.sklearn.log_model(
		sk_model=best_model,
		artifact_path='model',
		registered_model_name=mlflow_model_name
	)

	# Verify if the best model meets the desired score
	if test_score < desired_test_score:
		print(f"Testing score {test_score:.2f}% is still below the desired threshold of {desired_test_score:.2f}%.")
		sys.exit(-1)

	print("Found model with acceptable accuracy!")
	sys.exit(0)
