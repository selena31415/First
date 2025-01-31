# 手写数字识别 降维
from sklearn.decomposition import PCA
# 随机森林
from sklearn.ensemble import RandomForestClassifier
# 交叉验证
from sklearn.model_selection import cross_val_score
# 数据分析
import pandas as pd
# 科学计算
import numpy as np
# 可视化
import matplotlib.pyplot as plt

# 导入数据
data = pd.read_csv(r"D:\desktop\NeualNetworks\Tree\Skearn\3特征工程\digit_red.csv")
# 查看数据
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
# 查看数据
print(X.shape)
print(y.shape)


pca_line = PCA().fit(X)
pca_line.explained_variance_ratio_
# 绘制图像
plt.figure(figsize=(20, 5))
plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()