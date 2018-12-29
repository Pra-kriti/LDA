import numpy as np
# import matplotlib.pyplot as plt

X = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 5], [4, 2], [5, 0], [5, 2], [3, 2], [5, 3], [6, 3]])
y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

np.set_printoptions(precision=2)
class_labels = np.unique(y)
n_classes = class_labels.shape[0]
n_features = X.shape[1]


def calculate_mean_vectors(X, y):
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y == cl], axis=0))
        print('Mean vector for class %s: %s\n' % (cl, mean_vectors[cl - 1]))
    return mean_vectors


def between_scatter(X, y):
    total_mean = np.mean(X, axis=0)
    # print(total_mean)
    mean_vectors = calculate_mean_vectors(X, y)
    Sb = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i + 1, :].shape[0]
        u = mean_vec - total_mean
        u = np.matrix(u)
        Sb += n * np.multiply(np.transpose(u), u)
        print(i, Sb)
    print('\nbetween-class Scatter Matrix: \n', Sb)
    return Sb


def within_scatter(X, y):
    mean_vectors = calculate_mean_vectors(X, y)
    Sw = np.zeros((n_features, n_features))
    for cl, mv in zip(class_labels, mean_vectors):
        sc_class = np.zeros((n_features, n_features))
        for row in X[y == cl]:
            d = row - mv
            d = np.matrix(d)
            sc_class += np.multiply(np.transpose(d), d)
        Sw += sc_class
    print('\nwithin-class Scatter Matrix: \n', Sw)
    return Sw


def choose_features(eig_vals, eig_vecs, n_comp):
    # Make a list of (eigenvalue, eigenvectors) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvectors) tuples in descending order
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    # Visually confirming that the list is correctly sorted
    print(' \nEigen values in descending order:')
    for i in eig_pairs:
        print(i[0])

    W = np.hstack([eig_pairs[i][1].reshape(n_features, 1) for i in range(0, n_comp)])
    print('\nMatrix W: ', W.real)
    return W


Sw, Sb = within_scatter(X, y), between_scatter(X, y)
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
W = choose_features(eig_vals, eig_vecs, n_comp=1)
print('Eigen Value: %s\n\n Eigen Vectors: %s ' %(eig_vals, eig_vecs))


X_lda = X.dot(W)
print(X_lda)

