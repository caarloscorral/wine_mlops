import os
import json
import pickle
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Loading env variables
load_dotenv()

data_path = os.getenv('DATA_PATH')
model_path = os.getenv('MODEL_PATH')
metrics_path = os.getenv('METRICS_PATH')
desired_test_score = float(os.getenv('DESIRED_TEST_SCORE'))
model_seed = int(os.getenv('MODEL_SEED'))

################################
########## DATA PREP ###########
################################

# Loading in the data
df = pd.read_csv(data_path)

# Splitring into train and test sections
y = df.pop('quality')
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=model_seed)

#################################
########## MODELLING ############
#################################

# Trying to load best model
model_directory = os.path.dirname(model_path)
if not os.path.exists(model_directory):
	os.makedirs(model_directory, exist_ok=True)

if os.path.exists(model_path):
	print(model_path)
	with open(model_path, 'rb') as model_file:
		model = pickle.load(model_file)
	print("Loaded existing best model.")

# If no best model creating a new one
else:
	model = RandomForestRegressor(max_depth=2, random_state=model_seed)
	model.fit(X_train, y_train)
 
	# Saving new model
	with open(model_path, 'wb') as model_file:
		pickle.dump(model, model_file)
	
	print("Trained and saved a new model.")


#################################
########### METRICS #############
#################################

# Reporting train score
train_score = model.score(X_train, y_train)
print("Training variance explained: %2.1f%%" % train_score)

# Reporting test score
test_score = model.score(X_test, y_test)
print("Test variance explained: %2.1f%%" % test_score)

# Saving metrics to file
with open(metrics_path, 'w') as json_file:
    json.dump(
        {
			"Training variance explained": f"{train_score:.1f}",
			"Test variance explained": f"{test_score:.1f}"
		},
		json_file,
		indent=4
	)

print("Saved metrics to metrics.json")
