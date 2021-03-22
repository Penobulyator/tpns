import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def trim(values):
    x_25 = np.nanquantile(values, 0.25)
    x_75 = np.nanquantile(values, 0.75)

    lower_border = x_25 - 1.5 * (x_75 - x_25)
    higher_border = x_75 + 1.5 * (x_75 - x_25)

    for i in range(0, len(values)):
        if (values[i] < lower_border) | (values[i] > higher_border):
            values[i] = np.nan


def fill_gaps(values):
    nan_count = values.count(np.nan)
    if (nan_count != 0) & (nan_count / len(values) < 0.3):
        avg = np.nanmean(values)

        for i in range(0, len(values) - 1):
            if values[i] is np.nan:
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
    columns = list(corr_matrix.index)
    for i in columns:
        for j in columns[columns.index(i) + 1: len(columns) - 1]:

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
        #fill_gaps(arr)
        trim(arr)

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
    corr_matrix = dataframe.corr()
    analyze_correlations(corr_matrix)

    # build histograms
    hists = []
    plt.rc('figure', max_open_warning=0)
    for name in dataframe.columns:
        tmp = pd.DataFrame(dataframe[name])
        hists.append(tmp.hist())

    # show
    plt.show()


if __name__ == "__main__":
    main()
