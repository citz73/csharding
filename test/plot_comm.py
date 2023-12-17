import matplotlib.pyplot as plt
import argparse

# Define a function to read the data from the file and extract fw_comm_costs and bw_comm_costs
def read_data(filename):
    fw_comm_costs = []
    bw_comm_costs = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            _, _, fw_costs, bw_costs = map(lambda x: list(map(float, x.split(','))), values)
            fw_comm_costs.extend(fw_costs)
            bw_comm_costs.extend(bw_costs)
    return fw_comm_costs, bw_comm_costs

# Create an argument parser
parser = argparse.ArgumentParser(description='Generate a histogram of cost frequencies.')

# Add a positional argument for the filename
parser.add_argument('folder', type=int, help='Input CPU number')
# Parse the command-line arguments
args = parser.parse_args()

# Specify the file name
filename = f'../data/cost_data/comm_{args.folder}/data.txt'

# Read the data from the file
fw_comm_costs, bw_comm_costs = read_data(filename)

# Create histograms for fw_comm_costs and bw_comm_costs
plt.figure(figsize=(8, 6))


plt.hist(fw_comm_costs, bins=20, edgecolor='black')
plt.title(f'Distribution of Forward Communication Costs ({args.folder} CPU)')
plt.xlabel('Forward Communication Cost')
plt.ylabel('Frequency')
plt.savefig('plot/4fw_comm_costs_histogram.png')  # Save the plot as a PNG file
plt.clf()  # Clear the current figure

plt.hist(bw_comm_costs, bins=20, edgecolor='black')
plt.title(f'Distribution of Backward Communication Costs ({args.folder} CPU)')
plt.xlabel('Backward Communication Cost')
plt.ylabel('Frequency')
plt.savefig(f'plot/{args.folder}bw_comm_costs_histogram.png')  # Save the plot as a PNG file

