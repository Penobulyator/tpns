import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def rename_params(params):
    print("Renaming parameters:")
    start = ord('A')
    for i in range(0, len(params)):
        new_name = chr(start + i)
        if ord(new_name) >= ord('Z'):
            new_name = 'A' + chr(start + i % (ord('Z') - start))

        print('{} -> {}'.format(params[i], new_name))
        params[i] = new_name


def print_correlations(names, corr):
    for i in names:
        for j in names:
            if (i > j) & (corr[i][j] > 0.9):
                print('corr({}, {}) > 0.9'.format(i, j))


def read_csv(csvfile, names, values):
    data_reader = csv.reader(csvfile, delimiter=';')
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


def main():
    names = []
    values = []
    with open('dataset.csv', newline='') as csvfile:
        read_csv(csvfile, names, values)

    rename_params(names)

    # build data map
    data = {}
    for i in range(0, len(names)):
        data[names[i]] = values[i]

    df = pd.DataFrame(data)

    # build heat map
    corr_matrix = df.corr()
    print_correlations(names, corr_matrix)

    heatmap = sns.heatmap(corr_matrix, linewidths=.5, xticklabels=True, yticklabels=True)

    # build histograms
    for name in names:
        tmp = pd.DataFrame({name: data[name]})
        tmp.hist()

    plt.show()


if __name__ == "__main__":
    main()
