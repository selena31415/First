# 导入需要的库
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import process_time

# 导入数据，探索数据
data = load_breast_cancer()

# 将数据转换为 DataFrame
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['target'] = data.target
# 将 DataFrame 保存到 csv 文件中
# df.to_csv('breast_cancer_data.csv', index=False)

feature_names = ['平均半径', '平均纹理', '平均周长', '平均面积',
                 '平均平滑度', '平均紧密度', '平均凹度',
                 '平均凹点', '平均对称性', '平均分形维数',
                 '半径误差', '纹理误差', '周长误差', '面积误差',
                 '平滑度误差', '紧密度误差', '凹度误差',
                 '凹点误差', '对称性误差', '分形维数误差',
                 '最差半径', '最差纹理', '最差周长', '最差面积',
                 '最差平滑度', '最差紧密度', '最差凹度',
                 '最差凹点', '最差对称性', '最差分形维数']


def solve():
    rfc = RandomForestClassifier(n_estimators=100, random_state=90)
    # 交叉验证
    score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    print(score_pre)


def learn_curve():
    # 学习曲线
    scorel = []
    for i in range(0, 200, 10):
        rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1, random_state=90)
        score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
        scorel.append(score)
    # 输出最优参数
    print(max(scorel), scorel.index(max(scorel)) * 10 + 1)
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 201, 10), scorel)
    plt.show()


# 进一步细化学习曲线
def learn_curve2():
    # 学习曲线
    scorel = []
    for i in range(65, 75):
        rfc = RandomForestClassifier(n_estimators=i, n_jobs=-1, random_state=90)
        score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
        scorel.append(score)
    # 输出最优参数
    print(max(scorel), [*range(65, 75)][scorel.index(max(scorel))])
    plt.figure(figsize=[20, 5])
    plt.plot(range(65, 75), scorel)
    plt.show()


# 深度最好是8
def max_depth():
    # 调整参数 max_depth
    param_grid = {'max_depth': np.arange(1, 20, 1)}
    rfc = RandomForestClassifier(n_estimators=73, random_state=90)
    GS = GridSearchCV(rfc, param_grid, cv=10)
    GS.fit(data.data, data.target)
    print(GS.best_params_)
    print(GS.best_score_)

# 调整 max_features 24
def max_features():
    # 调整参数 max_features
    param_grid = {'max_features': np.arange(5, 30, 1)}
    rfc = RandomForestClassifier(n_estimators=73, random_state=90)
    GS = GridSearchCV(rfc, param_grid, cv=10)
    GS.fit(data.data, data.target)
    print(GS.best_params_)
    print(GS.best_score_)


if __name__ == '__main__':
    # 开始时间
    start = process_time()
    # 求解
    max_features()
    # 结束时间
    end = process_time()
    print("所用时间为{}".format(end - start))
