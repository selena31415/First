import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer  # 填补缺失值
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score

# 从本地导入波士顿房价数据集
boston = pd.read_csv(r'..\..\data\boston.csv')
# print(boston.columns)
# 划分数据和标签
X_full, y_full = boston.iloc[:, :-1], boston.iloc[:, -1]
n_samples = X_full.shape[0]
n_features = X_full.shape[1]
missing_rate = 0.5
n_missing_samples = int(np.floor(n_samples * n_features * missing_rate))
# 确定一个随机种子
rng = np.random.RandomState(0)
# 确定缺失值的位置
missing_features = rng.randint(0, n_features, n_missing_samples)
missing_samples = rng.randint(0, n_samples, n_missing_samples)
# np.choice 抽取不重复的数
X_missing = X_full.copy()
y_missing = y_full.copy()
for i in range(n_missing_samples):
    X_missing.iloc[missing_samples[i], missing_features[i]] = np.nan
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_missing_mean = imp_mean.fit_transform(X_missing)

X_missing_reg = X_missing.copy()
# argsort 带索引的排序
sortindex = np.argsort(X_missing_reg.isnull().sum()).values
# 填补缺失值
for i in sortindex:
    # 构建新的特征矩阵和新标签
    df = X_missing_reg
    fillc = df.iloc[:, i]
    df = pd.concat([df.iloc[:, df.columns != i], pd.DataFrame(y_full)], axis=1)
    # 在新特征矩阵中，对含有缺失值的列，进行0的填补
    df_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(df)
    # 找出训练集和测试集
    # Ytrain 是用来训练的特征标签，
    # Ytest 是用来测试的特征标签，
    # Xtrain 是用来训练的特征矩阵，
    # Xtest 是用来测试的特征矩阵
    Ytrain = fillc[fillc.notnull()]
    # Ytest 主要用其索引，
    Ytest = fillc[fillc.isnull()]
    Xtrain = df_0[Ytrain.index, :]
    Xtest = df_0[Ytest.index, :]
    # 用随机森林回归来填补缺失值
    rfc = RandomForestRegressor(n_estimators=100)
    rfc = rfc.fit(Xtrain, Ytrain)
    Ypredict = rfc.predict(Xtest)
    # 将填补好的特征返回到我们的原始的特征矩阵中
    X_missing_reg.iloc[X_missing_reg.iloc[:, i].isnull(), i] = Ypredict
# 查看四种填补方式的结果
X_misss_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(X_missing)
X = [X_full, X_missing_mean, X_misss_0, X_missing_reg]
mse = []
std = []
for x in X:
    estimator = RandomForestRegressor(random_state=0, n_estimators=100)
    scores = cross_val_score(estimator, x, y_full, scoring='neg_mean_squared_error', cv=10).mean()
    mse.append(scores * -1)
print("mse", mse)
# 用所得数据画图
x_labels = ['Full data', 'Mean Imputation', 'Zero Imputation', 'Regressor Imputation']
colors = ['r', 'g', 'b', 'orange']
plt.figure(figsize=(12, 6))
ax = plt.subplot(111)
for i in np.arange(len(mse)):
    ax.barh(i, mse[i], color=colors[i], alpha=0.6, align='center')
ax.set_title('Imputation Techniques with Boston Data')
ax.set_xlim(left=np.min(mse) * 0.9, right=np.max(mse) * 1.1)
ax.set_yticks(np.arange(len(mse)))
ax.set_xlabel('MSE')
ax.invert_yaxis()
ax.set_yticklabels(x_labels)
plt.show()
