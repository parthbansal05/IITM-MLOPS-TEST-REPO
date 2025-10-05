import pandas as pd

# Read the CSV file
df = pd.read_csv("iris.csv")

# Shuffle the rows
df = df.sample(frac=1, random_state=None).reset_index(drop=True)

# Save to a new CSV file
df.to_csv("shuffled_output.csv", index=False)
