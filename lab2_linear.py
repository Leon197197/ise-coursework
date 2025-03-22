from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="lab2 linear")
    parser.add_argument("--file", "-f",type=str, default="ambivert.csv", help="file name")
    parser.add_argument("--column", "-c",type=str, default="jobs", help="column name")
    args = parser.parse_args()

    param_file=args.file
    param_x = args.column

    systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
    for current_system in systems:
        datasets_location = 'datasets/{}'.format(current_system)  # Modify this to specify the location of the datasets
        colName = "time"
        if current_system == "h2":
            colName = "throughput"
        csv_files = [f for f in os.listdir(datasets_location) if
                     f.endswith('.csv')]  # List all CSV files in the directory
        # Initialize a dict to store results for average of the metrics

        for csv_file in csv_files:
            if csv_file == param_file:
                data = pd.read_csv(os.path.join(datasets_location, csv_file))
                plt.scatter(data[param_x], data[colName])
                plt.title(f"{csv_file}-{param_x}")
                #plt.scatter(data.index, data['RANK'])
                plt.xlabel(param_x)
                plt.ylabel(colName)
                plt.show()

if __name__ == "__main__":
    main()



