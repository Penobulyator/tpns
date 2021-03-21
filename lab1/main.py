import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def fill_gaps(values):
    if values.count(np.nan) / len(values) < 0.3:
        avg = np.nanmean(values)

        for i in range(0, len(values) - 1):
            if values[i] == np.nan:
                values[i] = avg


def rename_columns(dataframe):
    print("Renaming parameters:")
    start = ord('A')

    for i, column in enumerate(dataframe.columns):
        new_name = chr(start + i)
        if ord(new_name) >= ord('Z'):
            new_name = 'A' + chr(start + i % (ord('Z') - start))

        print('{} -> {}'.format(column, new_name))

        dataframe = dataframe.rename(columns={column: new_name})

    print("")

    return dataframe


def columns_correlate_similarly(col1, col2, corr_matrix):
    for column in corr_matrix:
        c1 = corr_matrix[col1][column]
        c2 = corr_matrix[col2][column]
        if abs(c1 - c2) > 0.3:
            return False

    return True


def analyze_correlations(corr_matrix):
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i >= j:
                continue

            corr = corr_matrix[i][j]
            if (corr > 0.9) & columns_correlate_similarly(i, j, corr_matrix):
                print("Drop {} or {}".format(i, j))


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

    for arr in values:
        fill_gaps(arr)

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
    analyze_correlations(corr_matrix)

    # build histograms
    for name in dataframe.columns:
        tmp = pd.DataFrame(dataframe[name])
        tmp.hist()

    # show
    plt.show()


if __name__ == "__main__":
    main()
