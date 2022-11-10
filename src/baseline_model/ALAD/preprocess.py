import numpy as np

data = np.load('../../data/DGraphFin/dgraphfin.npz')
features = data['x']
label = data['y']
features = features[label == (0 or 1)]
label = label[label == (0 or 1)]
print(len(features))
print(max(label))