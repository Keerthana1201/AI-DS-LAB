from pandas import *
from numpy import *
from sklearn.preprocessing import *
from sklearn.decomposition import *
from seaborn import *
from matplotlib.pyplot import *

data = DataFrame({
    'Age': [20, 19, 19, 21, 22],
    'Height': [160, 150, 185, 142, 130],
    'Weight': [55, 60, 75, 64, 75]
})

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

pca_df = DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])
print(pca_df)

cov_matrix = cov(scaled_data, rowvar=False)
print("Covariance Matrix\n", cov_matrix)

heatmap(data, annot=True, cmap="coolwarm")
title("Original Data")
show()

heatmap(pca_df, annot=True, cmap="coolwarm")
title("PCA Result")
show()
