import csv
import itertools

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
from sklearn.feature_selection import mutual_info_classif


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
            if np.isnan(values[i]):
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


# [1, 1, 2, 2, 3, 3, 3, nan] -> [1, 2, 3, nan]
def split_column(column):
    counter = Counter(column)
    result = {}
    for key, value in counter.items():
        if np.isnan(key):
            if np.nan in result.keys():
                result[np.nan] += 1
            else:
                result[np.nan] = 1
        else:
            result[key] = value
    return result


def get_classes(dataframe, targets, groups_count):
    targets_groups = []
    for i in range(len(targets)):
        target = targets[i]
        intervals = []

        if groups_count[i] == -1:
            # divide by unique values

            target_unique_values = split_column(dataframe[target].tolist()).keys()
            for value in target_unique_values:
                if np.isnan(value):
                    intervals.append(value)
                else:
                    intervals.append([value, value])
        else:
            # divide by intervals
            borders = splitter(dataframe[target], groups_count[i])
            for j in range(len(borders) - 1):
                intervals.append([borders[j], borders[j + 1]])

        targets_groups.append(intervals)

    classes = []
    for elem in itertools.product(targets_groups[0], targets_groups[1]):
            new_class = {
                targets[0]: elem[0],
                targets[1]: elem[1]
            }
            classes.append(new_class)

    return classes


def row_belongs_to_class(dataframe, row_index, gain_ratio_class):
    for target in gain_ratio_class:
        row_target_value = dataframe[target][row_index]
        value_range = gain_ratio_class[target]

        if not isinstance(value_range, list):
            if not np.isnan(row_target_value):
                return False
        else:
            if np.isnan(row_target_value):
                return False
            elif not value_range[0] <= row_target_value <= value_range[1]:
                return False

    return True


def get_row_classes(dataframe, classes):
    row_classes = []

    for row_number, index in enumerate(dataframe.index):
        has_class = False
        for i, _class in enumerate(classes):
            if row_belongs_to_class(dataframe, index, _class):
                row_classes.append(i)
                has_class = True
                break
        if not has_class:
            print("Class is unknown for row number " + str(row_number))
            row_classes.append(0)

    return row_classes


def print_gain_ratio(dataframe):
    targets = ['КГФ', 'G_total']
    kgf_groups_count = 1 + int(math.log(len(dataframe['КГФ']), 2))
    classes = get_classes(dataframe, targets, [kgf_groups_count, -1])

    row_classes = get_row_classes(dataframe, classes)

    mic = mutual_info_classif(dataframe.fillna(0), row_classes)

    for i, column in enumerate(dataframe.columns):
        print("{} - {}".format(column, round(mic[i], 2)))

    return


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

    row_classes = get_row_classes(dataframe, test)

    for i, class_number in enumerate(row_classes):
        print("Row {} belongs to class {}".format(i, class_number))


def main():
    # read dataframe from file
    with open('dataset.csv', newline='') as csv_file:
        dataframe = read_csv(csv_file)

    # print gain ratios
    print_gain_ratio(dataframe)

    # build heatmap
    corr_matrix = dataframe.corr()
    analyze_correlations(corr_matrix)
    # sns.heatmap(corr_matrix)

    # build histograms
    plt.rc('figure', max_open_warning=0)
    for name in dataframe.columns:
        pd.DataFrame(dataframe[name]).hist()

    # show
    plt.show()


if __name__ == "__main__":
    main()
