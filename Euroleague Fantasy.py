# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:06:46 2024

@author: A111115708
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


# Load the basketball stats file to inspect the contents
file_path = r'C:\Users\A111115708\OneDrive - Deutsche Telekom AG\Dokumente\AI Generative Training\Euroleague_G_Players_2024.csv'
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the data
df.head()

# Step 1: Filter the data for rounds 1 to 4 and for players with more than 10 minutes of playtime
filtered_data = df[df['ROUND'].between(1, 4)]
filtered_data_minutes = filtered_data[filtered_data['MIN'] > 10]

# Step 2: Map the roles to broader categories
role_mapping = {
    'SG': 'Guard', 'PG': 'Guard', 'PG/SG': 'Guard', 'SG/F': 'Guard',
    'F': 'Small Forward', 'SG/F': 'Small Forward'
}
filtered_data_minutes['ROLE'] = filtered_data_minutes['ROLE'].replace(role_mapping)

# Step 3: Normalize relevant columns using StandardScaler (optional, if normalization is needed)
columns_to_normalize = ['STOP', 'DF', 'PossTOT', 'OPPON OPP PTS', 'TO', 'DEF RTG']
scaler = StandardScaler()
filtered_data_minutes[columns_to_normalize] = scaler.fit_transform(filtered_data_minutes[columns_to_normalize]) * 100

# Step 4: Create the Final Defensive Index using selected variables
filtered_data_minutes['Final Defensive Index'] = (
    filtered_data_minutes['STOP'] +
    filtered_data_minutes['DF'] + 
    filtered_data_minutes['PossTOT'] - 
    filtered_data_minutes['OPPON OPP PTS'] - 
    filtered_data_minutes['TO'] +
    filtered_data_minutes['DEF RTG']
)

# Step 5: Scale the Final Defensive Index to a range of 0 to 100 using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 100))
filtered_data_minutes['Final Defensive Index (0-100)'] = scaler.fit_transform(filtered_data_minutes[['Final Defensive Index']])

# Step 6: Create a table showing the average defensive index per 'TM NAME' and 'ROLE' (after mapping)
avg_defensive_index_per_team_role = filtered_data_minutes.groupby(['TM NAME', 'ROLE'])['Final Defensive Index (0-100)'].mean().reset_index()

# Step 7: Pivot the table to have 'TM NAME' as rows, 'ROLE' (after mapping) as columns, and the average Final Defensive Index (0-100) as values
pivot_table = filtered_data_minutes.pivot_table(
    index='TM NAME', 
    columns='ROLE', 
    values='Final Defensive Index (0-100)', 
    aggfunc='mean'
).reset_index()

# Display the pivot table of average defensive index by team and role
print(pivot_table)

# Step 8: (Optional) Create a scatter plot between Final Defensive Index and VAL
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data_minutes['Final Defensive Index (0-100)'], filtered_data_minutes['VAL'], alpha=0.6)
plt.xlabel('Final Defensive Index (0-100)')
plt.ylabel('VAL')
plt.title('Scatter Plot of Final Defensive Index vs VAL')
plt.grid(True)
plt.show()


output_file_path = r'C:\Users\A111115708\OneDrive - Deutsche Telekom AG\Trainings\AI Generative Training\defensive_index_fantasy.csv'
pivot_table.to_csv(output_file_path, index=False)

# Provide the download link to the user
output_file_path
# Export the merged data frame to a CSV file
output_file_path = r'C:\Users\A111115708\OneDrive - Deutsche Telekom AG\Trainings\AI Generative Training\players_mvp_defensiveindex.csv'
filtered_data.to_csv(output_file_path, index=False)

############################################################ Evaluation ################################################################

# Calculate the average VAL for each player over the filtered data (rounds 1 to 4) and add it back to the original dataset
val_l4r = filtered_data_minutes.groupby('NAME')['VAL'].mean().reset_index()
val_l4r.rename(columns={'VAL': 'VAL_L4R'}, inplace=True)

# Merge the average VAL (VAL_L4R) back into the dataset while keeping the original 'VAL' column intact
df_with_val_l4r = df.copy()
df_with_val_l4r = pd.merge(df_with_val_l4r, val_l4r, on='NAME', how='left')

output_file_path = r'C:\Users\A111115708\OneDrive - Deutsche Telekom AG\Trainings\AI Generative Training\val_l4r.csv'
df_with_val_l4r.to_csv(output_file_path, index=False)




