"""1. 导入需要的库"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from time import process_time as pt

"""2. 加载数据集"""
wine = load_wine()

"""3. 划分数据集"""
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)


def Output():
    clf = DecisionTreeClassifier(random_state=0)
    rfc = RandomForestClassifier(random_state=0)
    """训练然后打分"""
    clf = clf.fit(Xtrain, Ytrain)
    rfc = rfc.fit(Xtrain, Ytrain)
    s1 = clf.score(Xtest, Ytest)
    s2 = rfc.score(Xtest, Ytest)
    print("决策树分数{}".format(s1))
    print("随机森林分数{}".format(s2))


def cross_validation():
    """交叉验证"""
    rfc_l = []
    clf_l = []
    for i in range(10):
        rfc = RandomForestClassifier(n_estimators=25)
        rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
        rfc_l.append(rfc_s)
        clf = DecisionTreeClassifier()
        clf_s = cross_val_score(clf, wine.data, wine.target, cv=10).mean()
        clf_l.append(clf_s)
    '''绘制图像'''
    plt.plot(range(1, 11), rfc_l, label="Random Forest")
    plt.plot(range(1, 11), clf_l, label="Decision Tree")
    plt.legend()
    plt.show()


def n_estimators():
    """n_estimators 的学习曲线"""
    superpa = []
    for i in range(200):
        rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1)
        rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
        superpa.append(rfc_s)
    print(max(superpa), superpa.index(max(superpa)))
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 201), superpa)
    plt.show()

if __name__ == '__main__':
    b = pt()
    n_estimators()
    e = pt()
    print("程序执行时间为{}".format(e - b))
