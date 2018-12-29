"""
1. Compute the d-dimensional mean vectors for the different classes from the dataset.

2. Compute the scatter matrices (in-between-class and within-class scatter matrix).

3. Compute the eigenvectors (ee1,ee2,...,eed) and corresponding eigenvalues (λλ1,λλ2,...,λλd) for the
    scatter matrices.

4. Sort the eigenvectors by decreasing eigenvalues and choose k eigenvectors with the largest
    eigenvalues to form a d×k dimensional matrix WW (where every column represents an eigenvector).

5. Use this d×k eigenvector matrix to transform the samples onto the new subspace.
    This can be summarized by the matrix multiplication: YY=XX×WW (where XX is a n×d-dimensional
    matrix representing the n samples, and yy are the transformed n×k-dimensional samples
    in the new subspace).
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


"""
Data matrix and classes
Two classes = w1(5x2) and w2(6x2)
"""
X = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 5], [4, 2], [5, 0], [5, 2], [3, 2], [5, 3], [6, 3]])
y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
class_labels = np.unique(y)
print(class_labels)
n_classes = class_labels.shape[0]
print(n_classes)
print(X.shape[1])


# Calculate mean of each class
np.set_printoptions(precision=2)
mean_vectors = []
for cl in set(y):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('Mean vector for class %s: %s\n' %(cl, mean_vectors[cl-1]))
# print(mean_vectors)


# Calculate between-class scatter matrix
total_mean = np.mean(X, axis=0)
# print(total_mean)
Sb = np.zeros((2, 2))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y==i+1, :].shape[0]
    u = mean_vec-total_mean
    print(type(u))
    u = np.matrix(u)
    print(type(u))
    Sb += n*np.multiply(np.transpose(u), u)
    print(i,Sb)
print('\nbetween-class Scatter Matrix: \n', Sb)


# Calculate within-class scatter matrix
Sw = np.zeros((2, 2))
for cl, mv in zip(set(y), mean_vectors):
    sc_class = np.zeros((2, 2))
    for row in X[y==cl]:
        d = row - mv
        d = np.matrix(d)
        sc_class += np.multiply(np.transpose(d), d)
    Sw += sc_class
print('\nwithin-class Scatter Matrix: \n', Sw)


# Solution of generalized eigen value problem for transformation matrix
Sw_inv = np.linalg.inv(Sw)
eig_vals, eig_vecs = np.linalg.eig(Sw_inv.dot(Sb))
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:, i].reshape(2, 1)
    print('\nEigen vector {}: \n{}'.format(i+1, eigvec_sc.real))
    print('Eigen value {:}: \n{:.2e}'.format(i+1, eig_vals[i].real))


# Selection of linear discriminants for new feature subspace - Those with highest eigen values
# Make a list of (eigenvalue, eigenvectors) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvectors) tuples in descending order
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
# Visually confirming that the list is correctly sorted
print(' \nEigen values in descending order:')
for i in eig_pairs:
    print(i[0])


# Variance as percentage
print('\nVariance explained:')
eigv_sum = sum(eig_vals)
for i, j in enumerate(eig_pairs):
    print('Eigenvalue {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))


# Choose k eigen vectors with largest eigenvalues
# print(eig_pairs)
W = np.hstack((eig_pairs[0][1]))
print('\nMatrix W: ', W.real)


# Transforming samples onto the new subspace
X_lda = X.dot(W)
print(X_lda)



# clf = LDA()
# print(clf.fit(X, y))
# print(clf.predict([[-0.8, -1]]))