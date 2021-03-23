import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math


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


def columns_correlate_similarly(col1, col2, corr_matrix):
    for column in corr_matrix:
        c1 = corr_matrix[col1][column]
        c2 = corr_matrix[col2][column]
        if abs(c1 - c2) > 0.2:
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

    # trim noise
    for arr in values:
        fill_gaps(arr)
        trim(arr)

    # build dataframe
    data = {}
    for i in range(0, len(names)):
        data[names[i]] = values[i]

    return pd.DataFrame(data)


def splitter(values, n):
    step = (max(values) - min(values)) / n
    out = []
    for i in range(n + 1):
        out.append(values.min() + i * step)
    return out


def split_column(column):
    # TODO
    pass


def get_classes(dataframe, targets, groups_count):
    targets_groups = []
    for i in range(len(targets)):
        target = targets[i]
        intervals = []

        if groups_count[i] == -1:
            # divide by unique values

            target_unique_values = np.unique(dataframe[target][~np.isnan(dataframe[target])])
            for value in target_unique_values:
                if value is np.nan:
                    intervals.append(np.nan)
                else:
                    intervals.append([value, value])
        else:
            # divide by intervals
            borders = splitter(dataframe[target], groups_count[i])
            for j in range(len(borders) - 1):
                intervals.append([borders[j], borders[j + 1]])

        targets_groups.append(intervals)

    classes = []

    for i in range(len(targets_groups[0])):
        for j in range(len(targets_groups[1])):
            new_class = {
                targets[0]: targets_groups[0][i],
                targets[1]: targets_groups[1][j]
            }
            classes.append(new_class)

    return classes


def row_belongs_to_class(dataframe, row_index, gain_ratio_class):
    for target in gain_ratio_class:
        row_target_value = dataframe[target][row_index]
        value_range = gain_ratio_class[target]

        if value_range is np.nan:
            if row_target_value is not np.nan:
                return False
        else:
            if row_target_value is np.nan:
                return False
            elif not value_range[0] <= row_target_value <= value_range[1]:
                return False

    return True


def gain_ratio(dataframe):
    targets = ['КГФ', 'G_total']
    kgf_groups_count = 1 + int(math.log(len(dataframe['КГФ']), 2))
    classes = get_classes(dataframe, targets, [kgf_groups_count, -1])

    for column in dataframe.columns:
        T = split_column(column)
        # TODO


def main():
    # read dataframe from file
    with open('dataset.csv', newline='') as csv_file:
        dataframe = read_csv(csv_file)

    # print gain ratios
    gain_ratio(dataframe)

    # build heatmap
    corr_matrix = dataframe.corr()
    analyze_correlations(corr_matrix)
    # sns.heatmap(corr_matrix)

    # build histograms
    # plt.rc('figure', max_open_warning=0)
    # for name in dataframe.columns:
    #     pd.DataFrame(dataframe[name]).hist()

    # show
    plt.show()


def get_classes_test():
    dataframe = pd.DataFrame(data={
        'x': [0, 1, 2, 3, 4, 5, 6],
        'y': [np.nan, np.nan, -1, -1, -2, -2, -2]
    })

    test = get_classes(dataframe, ['x', 'y'], [3, -1])

    for i, _class in enumerate(test):
        print("Class{}".format(i))
        print("x: {}".format(str(_class['x'])))
        print("y: {}".format(str(_class['y'])))
        print("")


if __name__ == "__main__":
    main()
