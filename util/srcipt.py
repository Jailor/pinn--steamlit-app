import csv

input_file = 'v_island_site1.csv'
output_file = '../v_island_site1_good.csv'

with open(input_file, 'r', newline='') as infile:
    reader = csv.reader(infile, delimiter=';')
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=',')
        for row in reader:
            writer.writerow(row)