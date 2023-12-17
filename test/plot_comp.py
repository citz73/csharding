import matplotlib.pyplot as plt
import argparse

# Function to read the data from the file and extract costs
def read_cost_data(filename):
    costs = []
    with open(filename, 'r') as file:
        for line in file:
            if "cost:" in line:
                cost = float(line.split("cost:")[1].strip())
                costs.append(cost)
    return costs

# Create an argument parser
parser = argparse.ArgumentParser(description='Generate a histogram of cost frequencies.')

# Add a positional argument for the filename
parser.add_argument('folder', type=int, help='Input CPU number')
# Parse the command-line arguments
args = parser.parse_args()

# Specify the file name
filename = f'../data/cost_data/compute_{args.folder}/data.txt'

# Read the cost data from the file
costs = read_cost_data(filename)

# Create a histogram for cost frequencies
plt.figure(figsize=(8, 6))
plt.hist(costs, bins=20, edgecolor='black')
plt.title(f'Distribution of Compute Cost Frequencies ({args.folder} CPU)')
plt.xlabel('Cost')
plt.ylabel('Frequency')
plt.savefig(f'plot/{args.folder}cost_histogram_comp.png')  # Save the plot as a PNG file

