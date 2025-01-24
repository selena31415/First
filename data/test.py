import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mping
import math

from multilayer_perceptron import MultilayerPerceptron

# 将csv 文件中的数据转化为图像
data = pd.read_csv('mnist_demo.csv')
numbers_to_display = 144
num_cells = math.ceil(math.sqrt(numbers_to_display))


def show_image(int: numbers_to_display):
    plt.figure(figsize=(10, 10))
    for plot_index in range(numbers_to_display):
        digit = data[plot_index:plot_index + 1].values
        digit_label = digit[0][0]  # 标签
        digit_pixels = digit[0][1:]  # 数据
        image_size = int(math.sqrt(digit_pixels.shape[0]))
        frame = digit_pixels.reshape((image_size, image_size))
        plt.subplot(num_cells, num_cells, plot_index + 1)
        plt.imshow(frame, cmap='Greys')
        plt.title(digit_label)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


# 训练模型 用百分之八十的数据做训练 剩下的做测试
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)
train_data = train_data.values
test_data = test_data.values
# 训练数据集个数
num_train_examples = 7000
# 象素数据
x_train = train_data[:num_train_examples, 1:]
x_test = test_data[:, 1:]
# 标签
y_train = train_data[:num_train_examples, [0]]
y_test = test_data[:, [0]]
layers = [784, 25, 10]
normalize_data = True
max_iterations = 300
# 步长
alpha = 0.1
mp = MultilayerPerceptron(x_train, y_train, layers, normalize_data)
# 训练模型
(thetas, costs) = mp.train(max_iterations, alpha)
plt.plot(range(len(costs)), costs)
plt.xlabel('Grident steps')
plt.ylabel('Cost')
plt.show()
# 测试训练结果
y_train_predictions = mp.predict(x_train)
# 测试集数据测试
y_test_predictions = mp.predict(x_test)
# 计算准确率
tp = np.sum(y_train_predictions == y_train) / y_train.shape[0] * 100
print('Train accuracy: %.2f%%' % tp)
test_p = np.sum(y_test_predictions == y_test) / y_test.shape[0] * 100
print('Test accuracy: %.2f%%' % test_p)
# 可视化预测结果
plt.figure(figsize=(24,20))
beg = 65;
end = 128
num_cell = math.ceil(math.sqrt(end - beg))
for i in range(beg, end):
    digit_label = y_test[i, 0]
    digit_pixels = x_test[i, :]
    predict_label = y_test_predictions[i, 0]
    image_size = int(math.sqrt(digit_pixels.shape[0]))
    frame = digit_pixels.reshape((image_size, image_size))
    color_map = "Greens" if digit_label == y_test_predictions[i] else "Reds"
    plt.subplot(num_cell, num_cell, i-64)
    plt.imshow(frame, cmap=color_map)
    if digit_label == predict_label:
        plt.title("wrong")
    else:
        plt.title("correct")
    plt.tick_params(axis='both', which='both', bottom=True, top=True, left=True, right=True, labelbottom=True,
                    labelleft=True)

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()