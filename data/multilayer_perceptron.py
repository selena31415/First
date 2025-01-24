import numpy as np
from utils.features import prepare_for_training
from utils.hypothesis import sigmoid, sigmoid_gradient


class MultilayerPerceptron:
    def __init__(self, data, labels, layers, normalize_data=False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data = data_processed  # 数据
        self.labels = labels  # 标签
        self.layers = layers  # 层次
        self.normalize_data = normalize_data
        self.thetas = MultilayerPerceptron.thetas_init(self.layers)

    def train(self, iterations, learningrate):
        # 参数 包含了最大迭代次数 和步长
        unrolled_thetas = MultilayerPerceptron.thetas_unroll(self.thetas)
        # 梯度下降
        (optimized_theta,cost_history) = MultilayerPerceptron.gradient_descent(self.data, self.labels, self.layers,
                                              unrolled_thetas, iterations, learningrate)
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta, self.layers)
        return self.thetas, cost_history
    @staticmethod
    def thetas_init(layers):
        # 先初始化权重矩阵
        num_layers = len(layers)
        thetas = {}
        for i in range(num_layers - 1):
            in_count = layers[i]
            out_count = layers[i + 1]
            thetas[i] = np.random.rand(out_count, in_count+1) * 0.05
        return thetas

    # @staticmethod
    def predict(self, data):
        # 预测
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]
        num_examples = data_processed.shape[0]
        # 前向传播 是利用训练好的矩阵 计算出预测结果
        predictions = MultilayerPerceptron.forward_propagation(data_processed, self.thetas, self.layers)
        return np.argmax(predictions, axis=1).reshape(num_examples, 1)
    @staticmethod
    def thetas_unroll(thetas):
        num_theta_layers = len(thetas)
        # 展开后的数组
        unroll_theta = np.array([])
        for i in range(num_theta_layers):
            # flatten 将多维数组降为一维数组 np.hstack  的参数类型为元组
            unroll_theta = np.hstack((unroll_theta, thetas[i].flatten()))
        return np.array(unroll_theta)

    @staticmethod
    # 还原权重矩阵
    def thetas_roll(unroll_thetas, layers):
        num_layers = len(layers)  # 层数
        thetas = {}
        unrolled_shift = 0
        for i in range(num_layers - 1):
            in_count = layers[i]
            out_count = layers[i + 1]
            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_width * thetas_height
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_theta_unrolled = unroll_thetas[start_index:end_index]
            thetas[i] = layer_theta_unrolled.reshape(thetas_height, thetas_width)
            unrolled_shift += thetas_volume
        return thetas

    @staticmethod
    def gradient_descent(data, labels, layers, unrolled_thetas, Iterations, LearningRate):
        optimized_theta = unrolled_thetas
        # 记录每次的损失
        cost_history = []
        for _ in range(Iterations):
            cost = MultilayerPerceptron.cost_function(data, labels,
                                                      MultilayerPerceptron.thetas_roll
                                                      (optimized_theta, layers), layers)
            cost_history.append(cost)
            theta_gradient = MultilayerPerceptron.gradient_step(data, labels, optimized_theta, layers)
            optimized_theta = optimized_theta - LearningRate * theta_gradient
        return optimized_theta, cost_history
    # 添加损失函数定义
    @staticmethod
    def cost_function(data, labels, thetas, layers):
        num_layers = len(layers)
        # 数据集内样本数量
        num_examples = data.shape[0]
        num_labels = layers[-1]
        # 前向传播走一次
        predictions = MultilayerPerceptron.forward_propagation(data, thetas, layers)
        # 制作标签
        bitwise_labels = np.zeros((num_examples, num_labels))
        for i in range(num_examples):
            bitwise_labels[i, labels[i][0]] = 1
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))
        # 计算损失
        cost = (-1 / num_examples) * (bit_set_cost + bit_not_set_cost)
        return cost

    @staticmethod
    def forward_propagation(data, thetas, layers):
        num_layers = len(layers)
        # 数据集内样本数量
        num_examples = data.shape[0]
        in_layer_activation = data
        # 逐层计算
        for i in range(num_layers - 1):
            theta = thetas[i]
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))
            in_layer_activation = out_layer_activation
        # 返回输出层结果
        return in_layer_activation[:, 1:]

    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        # 还原矩阵
        theta = MultilayerPerceptron.thetas_roll(optimized_theta, layers)
        # 反向传播
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(data, labels, theta, layers)
        thetas_rolled_gradients = MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)
        return thetas_rolled_gradients

    @staticmethod
    # 反向传播
    def back_propagation(data, labels, theta, layers):
        num_layers = len(layers)
        (num_examples, num_features) = data.shape
        num_labels_types = layers[-1]
        # 差异项计算
        deltas = {}
        # 初始化操作
        for layers_index in range(num_layers - 1):
            in_count = layers[layers_index]
            out_count = layers[layers_index + 1]
            deltas[layers_index] = np.zeros((out_count, in_count + 1))
        for example_index in range(num_examples):
            layers_input = {}
            layers_activations = {}
            layers_activation = data[example_index, :].reshape((num_features, 1))
            layers_activations[0] = layers_activation
            # 逐层计算
            for layers_index in range(num_layers - 1):
                layer_theta = theta[layers_index]
                layer_input = np.dot(layer_theta, layers_activation)
                layers_activation = np.vstack((np.array([[1]]), sigmoid(layer_input)))
                layers_input[layers_index + 1] = layer_input
                layers_activations[layers_index + 1] = layers_activation
            output_layer_activation = layers_activation[1:, :]
            delta = {}
            # 标签处理
            bitwise_labels = np.zeros((num_labels_types, 1))
            bitwise_labels[labels[example_index][0]] = 1
            # 计算输出层与真实值之间的误差
            delta[num_layers - 1] = output_layer_activation - bitwise_labels
            # 遍历循环
            for layers_index in range(num_layers - 2, 0, -1):
                layer_theta = theta[layers_index]
                next_delta = delta[layers_index + 1]
                layer_input = layers_input[layers_index]
                # 添加偏置项 ????
                layer_input = np.vstack((np.array((1)), layer_input))
                # 根据公式计算
                delta[layers_index] = np.dot(layer_theta.T, next_delta) * sigmoid_gradient(layer_input)
                # 过滤偏置参数
                delta[layers_index] = delta[layers_index][1:, :]
            for layers_index in range(num_layers - 1):
                layer_delta = np.dot(delta[layers_index + 1], layers_activations[layers_index].T)
                deltas[layers_index] += layer_delta
        for layer_index in range(num_layers - 1):
            deltas[layer_index] = deltas[layer_index] * (1 / num_examples)
        return deltas