from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
import pandas as pd
from utlis.tools import split


def demo():
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    print(pd.DataFrame(data))
    # 创建 MinMaxScaler 对象
    scaler = MinMaxScaler()
    scaler = scaler.fit(data)
    result = scaler.transform(data)
    print(result)
    result_ = scaler.fit_transform(data)
    print(result_)
    print(scaler.inverse_transform(result_))


data = pd.read_csv("data/Narrativedata.csv", index_col=0)


def view():
    Age = data.loc[:, "Age"].values.reshape(-1, 1)
    # 查看前20个数据
    # print(Age[:20])
    # 三种缺失值填充方式
    imp_mean = SimpleImputer()
    imp_median = SimpleImputer(strategy="median")
    imp_0 = SimpleImputer(strategy="constant", fill_value=0)
    imp_mean = imp_mean.fit_transform(Age)
    imp_median = imp_median.fit_transform(Age)
    imp_0 = imp_0.fit_transform(Age)
    # 查看填补结果
    print(imp_mean[:20])
    print("--" * 40)
    print(imp_median[:20])
    # 在这里我们使用中位数进行填补
    data.loc[:, "Age"] = imp_median
    print("-" * 80)
    print(data.info())


# def split(sym):
#     print(sym * 40)

def LabelEncode():
    """将文字标签转换成数据"""
    y = data.iloc[:, -1]
    le = LabelEncoder()
    le = le.fit(y)
    label_y = le.transform(y)
    print(label_y)
    split()
    print(le.classes_)
    split()
    data.iloc[:, -1] = label_y
    print(data.head(5))


def Encode():
    data_ = data.copy()
    """将文字标签转换成数据"""
    print(data_.head())
    split()
    print(OrdinalEncoder().fit(data.iloc[:, 1:]).categories_)
    split()
    data_.iloc[:, 1:] = OrdinalEncoder().fit_transform(data.iloc[:, 1:])
    print(data_.head())
    return data_


def OneHot(Data):
    """将文字标签转换成数据"""
    data_ = Data.copy()
    # iloc 根据整数索引获取数据
    x = data_.iloc[:, 1:-1]
    print(x.head())
    split()
    enc = OneHotEncoder(categories='auto').fit(x)
    result = enc.transform(x).toarray()
    print(result)
    split()
    print(enc.get_feature_names_out())
    split()
    print(data_.head())
    split()
    # 合并表格
    data_ = pd.concat([data_, pd.DataFrame(result)], axis=1)
    print(data_.head())
    split()
    # 删除原标签
    data_.drop(["Sex", "Embarked"], axis=1, inplace=True)
    print(data_.head())
    split()
    data_.columns = ["index", "Age", "Survived", "Female", "Male", "Embarked_C", "Embarked_Q", "Embarked_S"]
    print(data_.head())
    return data_


def Bin(Data):
    print("前五行\n", Data.head())
    data_ = Data.copy()
    # 去除索引
    data_.drop(["index"], axis=1, inplace=True)
    print("前五行\n", data_.head())
    x = data_.iloc[:, 0].values.reshape(-1, 1)
    t = Binarizer(threshold=30).fit_transform(x)
    print(t)
    split()


if __name__ == '__main__':
    # LabelEncode()
    # split()
    # print("OneHot-------------------")
    d = OneHot(Encode())
    print("Bin-------------------")
    Bin(d)
