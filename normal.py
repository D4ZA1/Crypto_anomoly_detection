import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.model_selection import train_test_split


df = pd.read_csv('dataset.csv')

numeric_columns = df.select_dtypes(include=[np.number]).columns
df = df[numeric_columns]


train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


model = hmm.GaussianHMM(n_components=3, covariance_type="full")


model.fit(train_data)


hidden_states = model.predict(test_data)

print(hidden_states)