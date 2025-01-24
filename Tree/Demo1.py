from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

wine = load_wine()
# 将数据集转换为DataFrame
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
# df['target'] = wine.target
features_zh = [
    '酒精',
    '苹果酸',
    '灰分',
    '灰分的碱度',
    '镁',
    '总酚',
    '黄酮类化合物',
    '非黄酮类酚',
    '原花青素',
    '颜色强度',
    '色调',
    '稀释葡萄酒的OD280/OD315',
    '脯氨酸'
]
# 划分训练集和测试集
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)


# print(Xtrain.shape)
# print(Xtest.shape)
# 建立模型
# random_state=30 随机种子
def test(depth: int):
    clf = tree.DecisionTreeClassifier(criterion="entropy"
                                      , random_state=30
                                      , splitter="random"
                                      , max_depth=depth
                                      , min_samples_leaf=10  # 每个叶子节点至少有10个样本
                                      , min_samples_split=10  # 少于则不允许分支
                                      )
    clf = clf.fit(Xtrain, Ytrain)
    # 返回正确率
    score = clf.score(Xtest, Ytest)
    print("正确率：", score)
    return score
    # # 画出决策树
    # dot_data = tree.export_graphviz(clf
    #                                 , feature_names=features_zh
    #                                 , class_names=["琴酒", "雪莉", "贝尔摩德"]
    #                                 , filled=True
    #                                 , rounded=True
    #                                 , fontname="Consolas,KaiTi"
    #                                 )
    # g = graphviz.Source(dot_data)
    # g.view()

scores = []
for _ in range(1, 20):
    scores.append(test(_))
# 绘制参数曲线
plt.plot(range(1, 20), scores)
plt.show()