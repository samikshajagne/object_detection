import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Load the Excel file
file_path = 'detections.xlsx'  # Update with your Excel file path
data = pd.read_excel(file_path)

# Display the first few rows of the dataframe
print(data.head())

# 1. Generate a bar chart for the 'Class' and 'Score'
plt.figure(figsize=(10, 6))
sns.barplot(x='Class', y='Score', data=data)
plt.title('Bar Chart of Object Classes with Scores')
plt.xlabel('Object Class')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Generate a pie chart for class distribution
plt.figure(figsize=(8, 8))
data['Class'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart of Class Distribution')
plt.ylabel('')
plt.show()

# 3. Create a map using GeoPandas (assuming you have latitude and longitude)
# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.Longitude, data.Latitude))

# Create a world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plot the points on the map
fig, ax = plt.subplots(figsize=(15, 10))
world.boundary.plot(ax=ax, linewidth=1)
gdf.plot(ax=ax, color='red', markersize=5, alpha=0.5)
plt.title('Map of Detected Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# You can add more visualizations based on your data insights
