from sklearn import svm
import numpy as np
import pandas as pd

'''
C: It is the regularization parameter, C, of the error term.
kernel: It specifies the kernel type to be used in the algorithm. It can be ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’, or a callable. The default value is ‘rbf’.
degree: It is the degree of the polynomial kernel function (‘poly’) and is ignored by all other kernels. The default value is 3.
gamma: It is the kernel coefficient for ‘rbf’, ‘poly’, and ‘sigmoid’. If gamma is ‘auto’, then 1/n_features will be used instead.

svm used for small+ high dimensional dataset, large dataset cause for long training time.

ovo-all set considered, one verus one
ovr-one versus rest, less comparisions, small set tent to be ignored


'''

data = pd.read_csv('./data.csv')

print(data.head(0))
print(data.columns.values[1])
print(data[[data.columns.values[1]]])


filter_data = data[['Flour','Sugar']]#.values is needed or not doesnt matter, as_matrix is the same as .values
print(type(filter_data))
label_creation = np.where(data['Type'] == 'Muffin', 0, 1)

print(label_creation)

model = svm.SVC(kernel='linear')
#SVC (C=1.0, kernel=’rbf’, degree=3, gamma=’auto’)
model.fit(filter_data, label_creation)
print(model.predict([[40,20]]))