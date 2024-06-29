import pandas as pd

# Assuming your CSV file is named 'data.csv'
input_csv_file = 'v_island_site1_good.csv'
output_csv_file = '../cleaned_v1_e350.csv'

# Read the CSV file
df = pd.read_csv(input_csv_file)

# Convert the Timestamp column to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m.%d.%y %H:%M')

# Save the cleaned DataFrame to a new CSV file (optional)
df.to_csv(output_csv_file, index=False)

# Now df contains the cleaned data and can be used for further processing
print(df.head())