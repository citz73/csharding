import matplotlib.pyplot as plt

# List of values and corresponding names
values = [
    23990.08661705613,
    27542.17448595047,
    16647.20842916315,
    27589.119325790405,
    32882.60566954255,
    26986.32693787098
]
names = [
    'dim_greedy',
    'lookup_greedy',
    'neuroshard',
    'random',
    'size_greedy',
    'size_lookup_greedy'
]

# Create a bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(names, values)
plt.title('16 CPU Sharding Latency Summary')
plt.xlabel('Sharding Algorithms')
plt.ylabel('Latency in ms')
plt.xticks(rotation=45)

# Add values on top of the bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}',
             ha='center', va='bottom')
    
# Set Y-axis limit to 35,000
plt.ylim(0, 85000)

plt.savefig(f'plot/16cost_result.png', bbox_inches='tight')  # Save the plot as a PNG file
