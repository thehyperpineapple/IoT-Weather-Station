import pandas as pd, numpy as np, matplotlib.pyplot as plt
import plotly.express as px
import os
import seaborn as sns

# Get the current working directory
current_directory = os.getcwd()

# Go back one level to the parent directory
parent_directory = os.path.dirname(current_directory)

# Open a different folder
desired_folder = "raw_data"  
folder_path = os.path.join(parent_directory, desired_folder)

# Check if the folder exists
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    file_to_read = "weatherHistory.csv" 
    file_path = os.path.join(folder_path, file_to_read)
    
    # Check if the file exists
    if os.path.exists(file_path) and os.path.isfile(file_path):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print(f"File '{file_to_read}' does not exist in folder '{desired_folder}'.")
else:
    print(f"Folder '{desired_folder}' does not exist in the parent directory.")

clean_df = df.drop(columns=["Formatted Date", "Daily Summary", "Visibility (km)", "Loud Cover","Wind Speed (km/h)", "Wind Bearing (degrees)", "Precip Type"])

clean_df = clean_df.drop(clean_df[clean_df['Summary'] == 'Dangerously Windy and Partly Cloudy'].index)

unwanted = ["Breezy and Overcast", "Breezy and Mostly Cloudy", "Breezy and Partly Cloudy", "Dry and Partly Cloudy","Windy and Partly Cloudy","Light Rain","Breezy", "Windy and Overcast", "Humid and Mostly Cloudy", "Dry and Mostly Cloudy", "Rain", "Windy", "Humid and Overcast", "Windy and Foggy", "Windy and Dry", "Breezy and Dry"]

for element in unwanted:
    clean_df = clean_df.drop(df[df['Summary'] == element].index)

df = clean_df

# Function to remove outliers using the IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out the outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply the function to each numerical column
numerical_columns = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Pressure (millibars)']
filtered_data = df.copy()

for column in numerical_columns:
    filtered_data = remove_outliers_iqr(filtered_data, column)

# Display the new shape of the dataset after outlier removal
filtered_data.shape

# Display the heatmap
plt.title('Correlation Matrix of Filtered Data')
plt.show()

clear_label_count = filtered_data[filtered_data['Summary'] == 'Clear'].shape[0]
p_cloudy_label_count = filtered_data[filtered_data['Summary'] == 'Partly Cloudy'].shape[0]
m_cloudy_label_count = filtered_data[filtered_data['Summary'] == 'Mostly Cloudy'].shape[0]
overcast_label_count = filtered_data[filtered_data['Summary'] == 'Overcast'].shape[0]

p_cloud_remove = p_cloudy_label_count - clear_label_count
m_cloudy_remove = m_cloudy_label_count - clear_label_count
overcast_remove = overcast_label_count - clear_label_count

# Specify the label and value you want to target
target_label = 'Summary'
target_value = 'Partly Cloudy'

# Step 2: Filter for specific label entries
target_entries = filtered_data[filtered_data[target_label] == target_value]

# Step 3: Randomly sample entries to remove (e.g., 2 entries)
entries_to_remove = target_entries.sample(n=p_cloud_remove)

# Step 4: Remove selected entries from the original DataFrame
filtered_data = filtered_data.drop(entries_to_remove.index)

filtered_data.to_csv('filtered_dataset.csv')


numerical_data = filtered_data.select_dtypes(include=[np.number])  

# Calculate the correlation matrix
correlation_matrix = numerical_data.corr(method='pearson')

# Use seaborn to create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)

# Display the heatmap
plt.title('Correlation Matrix of Filtered Data')
plt.show()