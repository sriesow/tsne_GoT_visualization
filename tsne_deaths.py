# Step 1: Import dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Step 2: Load the dataset
battles = pd.read_csv("dataset/character-deaths.csv")

# Step 3: Clean the seperate the dataset
label_data = battles["Gender"]
deaths_data = battles.drop(battles.columns[[0, 2, 3, 4, 5, 6]], axis=1).fillna("Nan")

# Step 5: Encode feature data
le = LabelEncoder()
for col in deaths_data.columns.values:
    if deaths_data[col].dtypes == "object":
        unique_labels = np.unique(deaths_data[col])
        le.fit(unique_labels)
        deaths_data[col] = le.transform(deaths_data[col])

# Step 6: Encode label data
le.fit(np.unique(label_data))
label_data = le.transform(label_data)

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(deaths_data)

# scatter plot the sample points among 5 classes
markers = ('s', 'd')
color_map = {0: 'red', 1: 'blue'}
plt.figure()
for idx, cl in enumerate(np.unique(label_data)):
    plt.scatter(x=x_test_2d[label_data == cl, 0], y=x_test_2d[label_data == cl, 1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of character deaths data')
plt.show()
