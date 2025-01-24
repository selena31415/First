"""两种正则化方法"""
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utlis.tools import split
data = load_breast_cancer()

X = data.data
Y = data.target
# print("X{}".format(X.shape))
# print("Y{}".format(Y))
lrl1 = LR(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)
lrl2 = LR(penalty="l2", solver="liblinear", C=0.5, max_iter=1000)

lrl1 = lrl1.fit(X, Y)
print(lrl1.coef_)
split()
print((lrl1.coef_!=0).sum(axis=1))
split()
lrl2 = lrl2.fit(X, Y)
split()
print(lrl2.coef_)
split()