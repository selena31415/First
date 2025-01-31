import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from time import process_time

file = r"../../../data/titanic/train.csv"
data = pd.read_csv(file)

# 数据预处理 axis=1  删除列
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# 增补缺失值
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna('S')
# 删除缺失值
data.dropna(inplace=True)
# 转换数据类型
labels = data['Embarked'].unique().tolist()
data['Embarked'] = data['Embarked'].apply(lambda x: labels.index(x))
# python 小技巧
data["Sex"] = (data["Sex"] == "male").astype("int")
x = data.iloc[:, data.columns != "Survived"]
y = data.iloc[:, data.columns == "Survived"]
# 划分数据集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3)
# 纠正索引
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])


# 跑模型
# clf = DecisionTreeClassifier(random_state=25)
# clf = clf.fit(Xtrain, Ytrain)
# score_ = clf.score(Xtest, Ytest)
# print(score_)
# # 使用交叉验证
# clf = DecisionTreeClassifier(random_state=25)
# score = cross_val_score(clf, x, y, cv=10).mean()
# print(score)
# 绘制学习曲线
def lerning_curve():
    tr = []
    te = []
    for i in range(10):
        clf = DecisionTreeClassifier(random_state=25
                                     , max_depth=i + 1
                                     , criterion="entropy"
                                     )
        clf = clf.fit(Xtrain, Ytrain)
        score_tr = clf.score(Xtrain, Ytrain)
        score_te = cross_val_score(clf, x, y, cv=10).mean()
        tr.append(score_tr)
        te.append(score_te)
    print(max(te))
    plt.plot(range(1, 11), tr, color="red", label="train")
    plt.plot(range(1, 11), te, color="blue", label="test")
    plt.xticks(range(1, 11))
    plt.legend()
    plt.show()


#  网格搜索
def grid_search():
    # * 的作用是将列表中的元素依次取出
    gini_threholds = np.linspace(0, 0.5, 50)
    parameters = {
        'criterion': ('gini', 'entropy')
        , 'splitter': ('best', 'random')
        , 'max_depth': [*range(1, 10)]
        , 'min_samples_leaf': [*range(1, 50, 5)]
        , 'min_impurity_decrease': [*np.linspace(0, 0.5, 50)]
    }
    clf = DecisionTreeClassifier(random_state=25)
    GS = GridSearchCV(clf, parameters, cv=10)
    GS = GS.fit(Xtrain, Ytrain)
    print(GS.best_params_)
    print(GS.best_score_)


if __name__ == '__main__':
    # 计算网络搜索所用的时间、
    start = process_time()
    grid_search()
    end = process_time()
    print("网络搜索所用的时间为：")
    print(end - start)

"""{'criterion': 'entropy', 'max_depth': 4, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 6, 'splitter': 'random'}
0.8104198668714797"""
