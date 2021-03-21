import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def rename_columns(dataframe):
    print("Renaming parameters:")
    start = ord('A')

    for i, column in enumerate(dataframe.columns):
        new_name = chr(start + i)
        if ord(new_name) >= ord('Z'):
            new_name = 'A' + chr(start + i % (ord('Z') - start))

        print('{} -> {}'.format(column, new_name))

        dataframe = dataframe.rename(columns={column: new_name})

    return dataframe


def print_correlations(corr):
    for i in corr.columns:
        for j in corr.columns:
            if (i > j) & (corr[i][j] > 0.9):
                print('corr({}, {}) > 0.9'.format(i, j))


def read_csv(csv_file):
    # read names and values
    names = []
    values = []
    data_reader = csv.reader(csv_file, delimiter=';')
    for i, row in enumerate(data_reader):
        for j, cell in enumerate(row):
            if i == 0:
                names.append(cell)
                values.append([])
            else:
                if cell != '':
                    values[j].append(float(cell))
                else:
                    values[j].append(np.nan)

    # build dataframe
    data = {}
    for i in range(0, len(names)):
        data[names[i]] = values[i]

    return pd.DataFrame(data)


def main():
    # read dataframe from file
    with open('dataset.csv', newline='') as csv_file:
        dataframe = read_csv(csv_file)

    # build heatmap
    corr_matrix = rename_columns(dataframe).corr()
    sns.heatmap(corr_matrix, linewidths=.5, xticklabels=True, yticklabels=True)

    # build histograms
    for name in dataframe.columns:
        tmp = pd.DataFrame(dataframe[name])
        tmp.hist()

    # show
    plt.show()


if __name__ == "__main__":
    main()
