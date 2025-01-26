import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
data = pd.read_csv('vixlarge.csv')

# Ensure the DATE column is in datetime format
data['DATE'] = pd.to_datetime(data['DATE'])

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(data['DATE'], data['VIX'], marker='o', linestyle='-', color='b', label='VIX Index')

# Label the axes and add a title
plt.xlabel('Date', fontsize=12)
plt.ylabel('VIX', fontsize=12)
plt.title('VIX Over Time', fontsize=14)
plt.grid(True)
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.tight_layout()
plt.show()