import os
import sys
import json
import pickle
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# Loading env variables
load_dotenv()

data_path = os.getenv('DATA_PATH')
model_path = os.getenv('MODEL_PATH')
metrics_path = os.getenv('METRICS_PATH')
desired_test_score = float(os.getenv('DESIRED_TEST_SCORE'))
rf_param_search = json.loads(os.getenv('RF_HYPERPARAM_SEARCH'))
model_seed = int(os.getenv('MODEL_SEED'))

# Loading data
df = pd.read_csv(data_path)

# Splitting data
y = df.pop('quality')
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=model_seed)

# Defining model
model = RandomForestRegressor(random_state=model_seed)

# Executing grid search
grid_search = GridSearchCV(
	estimator=model,
	param_grid=rf_param_search,
	cv=3,
	n_jobs=-1,
	verbose=0
)
grid_search.fit(X_train, y_train)

# Evaluating best model
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)

# Saving best model
with open(model_path, 'wb') as model_file:
	pickle.dump(best_model, model_file)

# Verifying if best model reaches desired accuracy threshold
if test_score < desired_test_score:
	print(f"Test score {test_score}% is still below the desired threshold of {desired_test_score}%.")
	sys.exit(-1)

print("Found model with acceptable accuracy!")
sys.exit(0)
