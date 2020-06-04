import math
import os, natsort as nsrt, numpy as np, re
from scipy.sparse import coo_matrix, csgraph, csr_matrix
import matplotlib.pyplot as plt
import cv2 as cv
import scipy
import sklearn
import argparse

from lib import models, graph, coarsening, utils

parser = argparse.ArgumentParser()
parser.add_argument("--path_project")
parser.add_argument("--path_data")
parser.add_argument("--zip_size")
parser.add_argument("--class_count")
parser.add_argument("--flag_save_zip", type=bool, default=False)
args = parser.parse_args()

PATH_PROJECT = args.path_project
PATH_DATA = args.path_data
zip_size = int(args.zip_size)
class_count = int(args.class_count)
flag_save_zip = bool(args.flag_save_zip)

PATH_GRAPHS = PATH_PROJECT + 'DATA/mini_graphs/graphs/'
list_grpahs = []
for (_, _, filenames) in os.walk(PATH_GRAPHS):
    list_grpahs = list_grpahs + filenames

list_grpahs = nsrt.natsorted(list_grpahs)[0::2]

num_samples = int(np.load(PATH_GRAPHS + list_grpahs[0])['num_samples'])
num_features = int(np.load(PATH_GRAPHS + list_grpahs[0])['num_features'])


def save_zip(save_size):
    list_of_rows = []
    list_of_cols = []
    list_of_max_vertices = []
    list_of_data = []

    zip_size = save_size

    for graph_name in list_grpahs:
        with np.load(PATH_GRAPHS + graph_name) as raw_graph:
            raw_edges = raw_graph['E'].transpose()
            rows = np.array(raw_edges[0])
            cols = np.array(raw_edges[1])

            max_range = max(np.max(rows), np.max(cols))
            unused_indexes = []
            for index in range(max_range):
                if (not index in rows) and (not index in cols):
                    unused_indexes.append(index)
            unused_indexes = np.array(unused_indexes)

            used_indexes = np.concatenate((rows, cols))
            used_indexes = np.unique(used_indexes, axis=0)
            used_indexes[::-1].sort()

            for used_var, unused_var in zip(used_indexes, unused_indexes):
                np.place(rows, rows == used_var, unused_var)
                np.place(cols, cols == used_var, unused_var)
            max_range = max(np.max(rows), np.max(cols))
            raw_data = raw_graph['D']

            list_of_rows.append(rows)
            list_of_cols.append(cols)
            list_of_max_vertices.append(max_range)
            list_of_data.append(raw_data)

            # print('used vertices shape: ', used_indexes.shape)
            # print('unused vertices shape:', unused_indexes.shape)
            # print('new max of vertices: ', max_range)

    assert np.max(list_of_max_vertices) == np.min(list_of_max_vertices)
    size_matrix = np.max(list_of_max_vertices) + 1

    X = []
    for raw_data, rows, cols in zip(list_of_data, list_of_rows, list_of_cols):
        sparse_graph = coo_matrix((raw_data, (rows, cols)),
                                  shape=(size_matrix, size_matrix))
        dense_graph = sparse_graph.todense()
        X.append(cv.resize(dense_graph,
                           dsize=(zip_size, zip_size),
                           interpolation=cv.INTER_CUBIC))
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

    PATH_LABELS = PATH_PROJECT + 'DATA/mini_graphs/GSE87571_samples.txt'

    raw_file = open(PATH_LABELS, 'r')
    y = []
    for line in raw_file.readlines():
        match_obj = re.match(r'(GSM[0-9]*)\s*([M,F])\s*([0-9]*)\s*([0-9]*)', line)
        if not match_obj is None:
            y.append(int(match_obj.group(3)))
    y = np.array(y)

    assert len(y) == num_samples
    assert len(X) == num_samples

    print(raw_graph.files)
    print(X.shape)
    print(y.shape)

    outfile = PATH_PROJECT + 'DATA/converted_data_resize_' + str(zip_size) + '.npz'
    np.savez(outfile, X, y)


if flag_save_zip:
    print('Start save zip graph matrix')
    save_zip(zip_size)

PATH_CONVERTED_DATA = PATH_PROJECT + 'DATA/' + PATH_DATA

npzfile = np.load(PATH_CONVERTED_DATA)
print(npzfile.files)
X = npzfile['arr_0'].astype(np.float32)
y = npzfile['arr_1']
print(X.shape)
print(y.shape)


def add_pad(X_full, zip_sz):
    X = []
    for graph in X_full:
        graph_size = int(math.sqrt(graph.shape[0]))
        new_graph = np.copy(graph).reshape(graph_size, graph_size)
        new_graph = np.pad(new_graph, pad_width=3, mode='constant', constant_values=0)
        X.append(cv.resize(new_graph,
                           dsize=(zip_sz, zip_sz),
                           interpolation=cv.INTER_CUBIC))
    X = np.array(X)
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    return X


# X = add_pad(X, zip_size)

from sklearn.utils import shuffle
X, y = shuffle(X, y)
##########################################################

print('--> Reshape data')
n_train = (num_samples * 4) // 5
n_val = num_samples // 20

X_train = X[:n_train]
X_val = X[n_train:n_train + n_val]
X_test = X[n_train + n_val:]

# 9 classes
if class_count == 9:
    y = y // 10 - 1

# 7 classes
if class_count == 7:
    y = y // 13 - 1

# 6 classes
if class_count == 6:
    y = y // 17

# 5 classes
if class_count == 5:
    y = y // 20

# 4 classes
if class_count == 4:
    y = y // 25

# 3 classes
if class_count == 3:
    y = y // 33

# 2 classes
if class_count == 2:
    y = y // 50

y_train = y[:n_train]
y_val = y[n_train:n_train + n_val]
y_test = y[n_train + n_val:]

plt.title("y = {}".format(y.shape[0]))
plt.hist(y, len(np.unique(y)))
plt.show()

plt.title("y_train = {}".format(y_train.shape[0]))
plt.hist(y_train, len(np.unique(y_train)))
plt.show()

plt.title("y_test = {}".format(y_test.shape[0]))
plt.hist(y_test, len(np.unique(y_test)))
plt.show()

print(np.unique(y))


##########################################################
def save_dump():
    print('--> Get distance graph')

    def distance_sklearn_metrics(z, k=4, metric='euclidean'):
        """Compute exact pairwise distances."""
        d = sklearn.metrics.pairwise.pairwise_distances(
            z, metric=metric, n_jobs=-2)
        # k-NN graph.
        idx = np.argsort(d)[:, 1:k + 1]
        d.sort()
        d = d[:, 1:k + 1]
        return d, idx

    dist, idx = distance_sklearn_metrics(X_train.T, k=4, metric='euclidean')
    A = graph.adjacency(dist, idx).astype(np.float32)

    PATH_DUMP_DATA = PATH_PROJECT + 'DATA/dump.npz'
    scipy.sparse.save_npz(PATH_DUMP_DATA, A)


save_dump()
PATH_DUMP_LOAD_DATA = PATH_PROJECT + 'DATA/dump.npz'
A = scipy.sparse.load_npz(PATH_DUMP_LOAD_DATA)

print('d = |V| = {}, k|V| < |E| = {}'.format(zip_size, A.nnz))
plt.spy(A, markersize=2, color='black')
plt.show()

print('--> Get laplacian matrix')
graphs, perm = coarsening.coarsen(A, levels=3, self_connections=True)
X_train = coarsening.perm_data(X_train, perm)
print(X_train.shape)
X_val = coarsening.perm_data(X_val, perm)
print(X_val.shape)
X_test = coarsening.perm_data(X_test, perm)
print(X_test.shape)

L = [graph.laplacian(A, normalized=True) for A in graphs]

params = dict()
params['dir_name'] = 'demo'
params['num_epochs'] = 32
params['batch_size'] = 16
params['eval_frequency'] = 100

# Building blocks.
params['filter'] = 'chebyshev5'
params['brelu'] = 'b1relu'
params['brelu'] = 'b2relu'
params['pool'] = 'apool1'
params['pool'] = 'mpool1'

# Number of classes.
C = y.max() + 1
assert C == np.unique(y).size

# Architecture.
params['F'] = [32, 32]  # Number of graph convolutional filters.
params['K'] = [16, 16]  # Polynomial orders.
params['p'] = [4, 2]  # Pooling sizes.
params['M'] = [2000, C]  # Output dimensionality of fully connected layers.

# Optimization.
params['regularization'] = 5e-4
params['dropout'] = 1
params['learning_rate'] = 1e-3
params['decay_rate'] = 0.95
params['momentum'] = 0
params['decay_steps'] = n_train / params['batch_size']

model = models.cgcnn(L, **params)
accuracy, loss, t_step = model.fit(X_train, y_train, X_val, y_val)

fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.plot(accuracy, 'b.-')
ax1.set_ylabel('validation accuracy', color='b')
ax2 = ax1.twinx()
ax2.plot(loss, 'g.-')
ax2.set_ylabel('training loss', color='g')
plt.show()

print('Time per step: {:.2f} ms'.format(t_step * 1000))

print(X_test.shape, y_test.shape)
acc_per_class = {}
for id_class in np.unique(y):
    acc_per_class[id_class] = []

for graph, label in zip(X_test, y_test):
    acc_per_class[label].append(graph)

for id_class in np.unique(y):
    acc_per_class[id_class] = np.array(acc_per_class[id_class])
    acc_shape = acc_per_class[id_class].shape
    labels = np.empty(acc_shape[0])
    labels.fill(id_class)
    print("############ Class {}".format(id_class))
    print(acc_shape)
    print(model.evaluate(acc_per_class[id_class], labels)[0])

print("############ All")
res = model.evaluate(X_test, y_test)
print(res[0])
