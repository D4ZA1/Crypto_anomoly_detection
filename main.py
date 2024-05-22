import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import cvxpy as cp


df = pd.read_csv('dataset.csv')
df.drop("timestamp", axis=1, inplace=True)
df.drop('sl', axis=1, inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df.set_index('date', inplace=True)
dfs = {name: group for name, group in df.groupby('crypto_name')}

class HMMModel:
    def __init__(self, n_components):
        self.model = hmm.GaussianHMM(n_components=n_components)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def retrain(self, X):
        self.model.init_params = ''
        self.model.fit(X)

class ADMM:
    def __init__(self, rho, alpha):
        self.rho = rho
        self.alpha = alpha

    def update(self, x, z, u):
        z_var = cp.Variable(z.shape)
        objective = cp.Minimize(cp.norm(x - z_var, 2) ** 2 + self.alpha * cp.norm(z_var, 1))
        constraints = []
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, max_iters=3, eps=1e-3, warm_start=True)
        print(f"Problem status: {problem.status}")  # Print the status of the problem
        z = z_var.value
        u = u + self.rho * (x - z)
        return z, u

hmm_models = {name: HMMModel(n_components=2) for name in dfs.keys()}
admm = ADMM(rho=1.0, alpha=0.1)


threshold = 0.01

for name, group in dfs.items():
    print(f"Processing {name}...")
    numeric_columns = group.select_dtypes(include=[np.number]).columns
    group_numeric = group[numeric_columns]

    group_resampled = group_numeric.resample('D').mean()


    group_interpolated = group_resampled.interpolate()


    group_differenced = group_interpolated.diff().dropna()


    group_normalized = (group_differenced - group_differenced.min()) / (group_differenced.max() - group_differenced.min())

    dfs[name]= group_normalized
    print(f"Data for {name} normalized.")

    hmm_models[name].fit(group_normalized.values)
    print(f"HMM model fitted for {name}.")

    z = np.zeros_like(group_normalized.values)
    u = np.zeros_like(z)
    for i in range(10):
        print(i)
        z, u = admm.update(group_normalized.values, z, u)
        if i % 10 == 0:
            print(f"ADMM update iteration {i} completed for {name}.")

    print(f"ADMM update completed for {name}.")

    hmm_models[name].retrain(z)
    print(f"HMM model retrained with optimized parameters for {name}.")


    predicted_states = hmm_models[name].predict(group_normalized.values)


    state_counts = np.bincount(predicted_states)
    total_count = len(predicted_states)
    state_frequencies = state_counts / total_count


    anomalous_states = np.where(state_frequencies < threshold)[0]


    anomaly_mask = np.isin(predicted_states, anomalous_states)

    final_group_values = group_normalized.values

    anomalies = final_group_values[anomaly_mask]

    flattened_values = final_group_values.flatten()

    flattened_anomalies = anomalies.flatten()


    anomaly_indices = np.array(np.where(anomaly_mask)).flatten()

    min_length = min(len(anomaly_indices), len(flattened_anomalies))
    anomaly_indices = anomaly_indices[:min_length]
    flattened_anomalies = flattened_anomalies[:min_length]

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(flattened_values)), flattened_values, color='b', label='Normal')
    plt.scatter(anomaly_indices, flattened_anomalies, color='r', label='Anomaly')
    plt.legend()
    plt.title(f"Anomalies in {name}")
    plt.show()