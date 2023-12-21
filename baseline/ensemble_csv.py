import pandas as pd

# Read in the csv files
df1 = pd.read_csv('output_1.csv')
df2 = pd.read_csv('output_2.csv')
df3 = pd.read_csv('output_3.csv')
df4 = pd.read_csv('output_4.csv')
df5 = pd.read_csv('output_5.csv')
df6 = pd.read_csv('output_6.csv')

# take first column from df1
df_temp = df1.iloc[:, 0]

# drop the first column of each csv file
df1 = df1.drop(columns=['id'])
df2 = df2.drop(columns=['id'])
df3 = df3.drop(columns=['id'])
df4 = df4.drop(columns=['id'])
df5 = df5.drop(columns=['id'])
df6 = df6.drop(columns=['id'])

# Merge the csv files
df = pd.concat([df1, df2, df3, df4, df5, df6], axis=1)

# Take the average of the predictions
df['target'] = df.mean(axis=1)

# drop the other columns
df = df.iloc[:, -1]

# Add the first column back
df = pd.concat([df_temp, df], axis=1)

# Save the csv file
df.to_csv('ensemble_output.csv', index=False)