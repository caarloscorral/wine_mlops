import os
import sys
import pickle
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Loading env variables
load_dotenv()

data_path = os.getenv('DATA_PATH')
model_path = os.getenv('MODEL_PATH')
metrics_path = os.getenv('METRICS_PATH')
desired_test_score = float(os.getenv('DESIRED_TEST_SCORE'))
model_seed = int(os.getenv('MODEL_SEED'))

# Loading model
with open(model_path, 'rb') as model_file:
    regr = pickle.load(model_file)
    
# Loading data
df = pd.read_csv(data_path)

# Splitting data
y = df.pop('quality')
_, X_test, _, y_test = train_test_split(df, y, test_size=0.2, random_state=model_seed)

# Evaluating model
test_score = regr.score(X_test, y_test)

# Checking if test score is lower than desired
if test_score < desired_test_score:
    # Returning error code if test score is lower than desired, to indicate there is needed a retraining
	print(f"Test score {test_score}% is below the desired threshold of {desired_test_score}%!")
	sys.exit(-1)

# Returning 0 if test score is higher than desired
print("Model meets the required accuracy!")
sys.exit(0)
