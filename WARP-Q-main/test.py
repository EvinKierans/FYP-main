import pandas as pd

training_data_df = pd.read_csv("Results.csv", header=0)
training_data_df = training_data_df.T
print(training_data_df[0])
training_data_df.to_csv("Results_check.csv")