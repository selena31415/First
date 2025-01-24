import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from utlis.tools import split
import pandas as pd
import numpy as np
# 提取数据集
iris = load_iris()
y = iris.target
x = iris.data
# print(pd.DataFrame(x))

# # 建模
# pca = PCA(n_components=2)
# pca = pca.fit(x) # 拟合模型
# x_dr = pca.transform(x) # 获取新矩阵
# # print(x_dr)
# # 可视化
# colors = ['red', 'black', 'orange']
# print("标签名", iris.target_names)
# split()
# plt.figure()
# for i in range(3):
#     plt.scatter(x_dr[y == i, 0], x_dr[y == i, 1], alpha=.7, c=colors[i], label=iris.target_names[i])
# plt.legend()  # 显示图例
# plt.title('PCA of IRIS dataset')
# plt.show()

# print("可解释的方差比例", pca.explained_variance_ratio_)
# print("可解释的方差", pca.explained_variance_)
# print("???", pca.explained_variance_ratio_.sum())

pca_line = PCA().fit(x)
plt.plot([1, 2, 3, 4], np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1, 2, 3, 4])  # 这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()