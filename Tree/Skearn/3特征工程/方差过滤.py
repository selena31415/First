# 方差过滤对结果的影响
# 方差过滤对结果的影响
# KNN vs 随机森林在不同方差过滤效果下的对比
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold  # 导入VarianceThreshold类
import numpy as np
import pandas as pd

data = pd.read_csv(r'D:\desktop\NeualNetworks\Tree\Skearn\3特征工程\digit_red.csv')
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)

# 随机森林在过滤后的数据集上的效果
cross_val_score(RFC(n_estimators=10, random_state=0), X_fsvar, y, cv=5).mean()
