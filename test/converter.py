import csv

# Input and output file paths
input_file_path = '/mnt/c/Users/tbjag/Downloads/data.txt'
output_file_path = 'output.csv'

# Open input file for reading and output file for writing
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w', newline='') as output_file:
    # Create a CSV writer object
    csv_writer = csv.writer(output_file)

    # Write header to CSV file
    csv_writer.writerow(['Task', 'Cost'])

    # Iterate through each line in the input file
    for line in input_file:
        # Split the line based on '|' and ',' to extract task numbers and cost
        parts = line.strip().split('|')
        task_numbers = [task.strip() for task in parts[0].replace('task:', '').split(',')]
        cost = parts[1].replace('cost:', '').strip()

        # Write the extracted data to the CSV file
        csv_writer.writerow([','.join(task_numbers), cost])

print(f"Conversion complete. CSV file saved at: {output_file_path}")
