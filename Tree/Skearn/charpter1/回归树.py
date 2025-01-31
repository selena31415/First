import numpy as np
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

# 生成随机数种子
rng = np.random.RandomState(1)
# 生成训练数据
X = np.sort(5 * rng.rand(80, 1), axis=0)
# ravel()函数将数组降维（每次降一维）
y = np.sin(X).ravel()
# 增加噪声 每五个数据点增加一个噪声
y[::5] += 3 * (0.5 - rng.rand(16))
# 拟合回归模型
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
a = regr_1.fit(X, y)
b = regr_2.fit(X, y)

# 预测 newaxis 升维
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
# 绘制结果
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
