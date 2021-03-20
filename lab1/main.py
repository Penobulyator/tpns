import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

names = []
values = []
with open('dataset.csv', newline='') as csvfile:
    dataReader = csv.reader(csvfile, delimiter=';')
    for i, row in enumerate(dataReader):
        for j, cell in enumerate(row):
            if (i == 0):
                names.append(cell)
                values.append([])
            else:
                if cell != '':
                    values[j].append(float(cell))
                else:
                    values[j].append(np.nan)

# build data map
df = {}
for i in range(0, len(names)):
    df[names[i]] = values[i]

# build heat map
heatmap_plot = sns.heatmap(values)


# build histograms
hists = []
for name in names:
    tmp = pd.DataFrame({name: df[name]})
    hists.append(tmp.hist())

plt.show()
