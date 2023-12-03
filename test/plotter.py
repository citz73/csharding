import matplotlib.pyplot as plt
import pandas as pd

# Read data from CSV
df = pd.read_csv('output.csv')  # Replace 'your_file.csv' with your actual file path

print(df.shape)

# Histogram
plt.hist(df['Cost'], bins=10, edgecolor='black')  # Adjust the number of bins as needed

# Adding labels and title
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.title('Distribution of Cost 4 vCPUs')

# Save the plot as a PNG file
plt.savefig('cost_distribution_plot.png')

# Display the plot
plt.show()

