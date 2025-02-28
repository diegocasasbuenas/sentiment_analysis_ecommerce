import pandas as pd

# Import dataset
data = pd.read_csv("petsentiment_analysis/data/raw/data_1000.csv" , parse_dates = ["timestamp"])

# Convert timestamp to datetime and filter the year 2023
print(data["timestamp"].dtype)

data_2023 = data[data['timestamp'].dt.year == 2023]

print(data["timestamp"].dtype) 

# Number of samples per rating
num_samples = 60493

# Create the balanced dataset using groupby and sample
balanced_data = data_2023.groupby('rating').apply(lambda x: x.sample(n=num_samples, random_state=42, replace=False)).reset_index(drop=True)

#Deleting null  values
balanced_data = balanced_data.dropna()

print(f"Number of raws after balance the dataframe and drop null values: {balanced_data.shape[0]}")

# Save the balanced dataset to a new CSV file
balanced_data.to_csv("petsentiment_analysis/data/raw/balanced_data.csv", index=False)


                       

