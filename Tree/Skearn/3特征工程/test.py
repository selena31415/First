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

# 卡方过滤
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2

X_fschi = SelectKBest(chi2, k=340).fit_transform(X_fsvar, y)

X_fschi.shape

# 交叉验证
from sklearn.model_selection import cross_val_score
import numpy as np

cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()

# %matplotlib inline
# 绘制学习曲线
import matplotlib.pyplot as plt

score = []
for i in range(390, 200, -10):
    X_fschi = SelectKBest(chi2, k=i).fit_transform(X_fsvar, y)
    once = cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()
    score.append(once)
plt.plot(range(390, 200, -10), score)
plt.show()

chivalue, pvalues_chi = chi2(X_fsvar, y)
chivalue
pvalues_chi
k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
k

# F检验 数据服从标准正态分布效果会比较好
from sklearn.feature_selection import f_classif

F, pvalues_f = f_classif(X_fsvar, y)
k = F.shape[0] - (pvalues_f > 0.05).sum()
k
# 互信息法
from sklearn.feature_selection import mutual_info_classif as MIC

result = MIC(X_fsvar, y)
k = result.shape[0] - (result <= 0).sum()
k

# 嵌入法
# 缺点：运行速度慢
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC

# 实例化随机森林
RFC_ = RFC(n_estimators=10, random_state=0)
X_embedded = SelectFromModel(RFC_, threshold=0.005).fit_transform(X, y)

# 绘制学习曲线找到最佳阈值
import numpy as np
import matplotlib.pyplot as plt

RFC_.fit(X, y).feature_importances_
threshold = np.linspace(0, (RFC_.fit(X, y).feature_importances_).max(), 20)
score = []
for i in threshold:
    X_embedded = SelectFromModel(RFC_, threshold=i).fit_transform(X, y)
    once = cross_val_score(RFC_, X_embedded, y, cv=5).mean()
    score.append(once)
plt.plot(threshold, score)
plt.show()

# RFE
from sklearn.feature_selection import RFE

RFC_ = RFC(n_estimators=10, random_state=0)
selector = RFE(RFC_, n_features_to_select=340, step=50).fit(X, y)
selector.support_.sum()
selector.ranking_
X_wrapper = selector.transform(X)
cross_val_score(RFC_, X_wrapper, y, cv=5).mean()

# 绘制包装法的学习曲线
score = []
for i in range(1, 751, 50):
    X_wrapper = RFE(RFC_, n_features_to_select=i, step=50).fit_transform(X, y)
    once = cross_val_score(RFC_, X_wrapper, y, cv=5).mean()
    score.append(once)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 751, 50), score)
plt.xticks(range(1, 751, 50))
plt.show()
