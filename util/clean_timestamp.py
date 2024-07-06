import pandas as pd


input_csv_file = '../old_data/rossland_original.csv'
output_csv_file = '../datasets/rossland_original.csv'

df = pd.read_csv(input_csv_file, low_memory=False)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')


df.to_csv(output_csv_file, index=False)

print(df.head())