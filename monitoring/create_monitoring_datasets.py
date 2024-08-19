import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original dataset
df = pd.read_csv("../project-mlops/data/hour.csv")


# Split the data into reference (older data) and production (newer data)
reference_data, production_data = train_test_split(df, test_size=0.3, shuffle=False)

# Save the datasets
reference_data.to_csv("reference_data.csv", index=False)
production_data.to_csv("production_data.csv", index=False)

print("Reference and production datasets created successfully.")
