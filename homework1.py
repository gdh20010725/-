from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tqdm._tqdm import trange
import cv2
from sklearn.decomposition import PCA
import random

import json

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
# 构造全连接层，实现 X 和 W 的矩阵乘法运算
mnist = input_data.read_data_sets("dataset/", one_hot=True)

class FullConnectionLayer():
    def __init__(self):
        self.m = {}

    def forward(self, X, W):

        self.m["X"] = X
        self.m["W"] = W
        H = np.matmul(self.m["X"], self.m["W"])
        return H

    def backward(self, grad_H):

        grad_X = np.matmul(grad_H, self.m["W"].T)
        grad_W = np.matmul(self.m["X"].T, grad_H) + 0.001 * self.m["W"]
        return grad_X, grad_W

# 实现 Leaky_Relu 激活函数
class Leaky_Relu():
    def __init__(self):
        self.m = {}

    def forward(self, X):
        self.m["X"] = X
        return np.where(X > 0, X, np.zeros_like(X))

    def backward(self, grad_y):
        X = self.m["X"]
        return np.where(X > 0, grad_y, 0.01 * grad_y)

# 交叉熵损失
class CrossEntropy():
    def __init__(self):
        self.m = {}
        self.epsilon = 1e-12

    def forward(self, p, y):
        self.m['p'] = p
        log_p = np.log(p + self.epsilon)
        return np.mean(np.sum(-y * log_p, axis=1))

    def backward(self, y):
        p = self.m['p']
        return -y * (1 / (p + self.epsilon))


# 实现 Softmax 激活函数
class Softmax():
    def __init__(self):
        self.m = {}
        self.epsilon = 1e-12

    def forward(self, p):
        p_exp = np.exp(p)
        denominator = np.sum(p_exp, axis=1, keepdims=True)
        s = p_exp / (denominator + self.epsilon)
        self.m["s"] = s
        self.m["p_exp"] = p_exp
        return s

    def backward(self, grad_s):
        s = self.m["s"]
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))
        tmp = np.matmul(np.expand_dims(grad_s, axis=1), sisj)
        tmp = np.squeeze(tmp, axis=1)
        grad_p = -tmp + grad_s * s
        return grad_p

# 搭建全连接神经网络模型
class FullConnectionModel():
    def __init__(self, latent_dims):
        self.W1 = np.random.normal(loc=0, scale=1, size=[28 * 28 + 1, latent_dims]) / np.sqrt((28 * 28 + 1) / 2)  # He 初始化，有效提高 Relu 网络的性能
        self.W2 = np.random.normal(loc=0, scale=1, size=[latent_dims, 10]) / np.sqrt(latent_dims / 2)  # He 初始化，有效提高 Relu 网络的性能
        self.mul_h1 = FullConnectionLayer()
        self.relu = Leaky_Relu()
        self.mul_h2 = FullConnectionLayer()
        self.softmax = Softmax()
        self.cross_en = CrossEntropy()

    def forward(self, X, labels):
        bias = np.ones(shape=[X.shape[0], 1])
        X = np.concatenate([X, bias], axis=1)
        self.h1 = self.mul_h1.forward(X, self.W1)
        self.h1_relu = self.relu.forward(self.h1)
        self.h2 = self.mul_h2.forward(self.h1_relu, self.W2)
        self.h2_soft = self.softmax.forward(self.h2)
        self.loss = self.cross_en.forward(self.h2_soft, labels)

    def backward(self, labels):
        self.loss_grad = self.cross_en.backward(labels)
        self.h2_soft_grad = self.softmax.backward(self.loss_grad)
        self.h2_grad, self.W2_grad = self.mul_h2.backward(self.h2_soft_grad)
        self.h1_relu_grad = self.relu.backward(self.h2_grad)
        self.h1_grad, self.W1_grad = self.mul_h1.backward(self.h1_relu_grad)


# 计算精确度
def computeAccuracy(prob, labels):
    predicitions = np.argmax(prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predicitions == truth)


# 训练一次模型
def trainOneStep(model, x_train, y_train, learning_rate=1e-5):
    model.forward(x_train, y_train)
    model.backward(y_train)
    model.W1 += -learning_rate * model.W1_grad
    model.W2 += -learning_rate * model.W2_grad
    loss = model.loss
    accuracy = computeAccuracy(model.h2_soft, y_train)
    return loss, accuracy


# 训练模型和寻优
def train(x_train, y_train, x_validation, y_validation):
    epochs = 200
    learning_rate = 1e-5
    latent_layer_list = [100, 200, 300, 400]
    best_accuracy = 0
    best_latent_dims = 0

    # 在验证集上寻找最优解
    print("Start seaching the best parameter...\n")
    for latent_dims in latent_layer_list:
        model = FullConnectionModel(latent_dims)

        bar = trange(20)
        for epoch in bar:
            loss, accuracy = trainOneStep(model, x_train, y_train, learning_rate)
            bar.set_description(f'Parameter latent_dims={latent_dims: <3}, epoch={epoch + 1: <3}, loss={loss: <10.8}, accuracy={accuracy: <8.6}')  # 给进度条加个描述
        bar.close()

        validation_loss, validation_accuracy = evaluate(model, x_validation, y_validation)
        print(f"Parameter latent_dims={latent_dims: <3}, validation_loss={validation_loss}, validation_accuracy={validation_accuracy}.\n")

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_latent_dims = latent_dims

    # 得到最好的参数组合，训练最好的模型
    print(f"The best parameter is {best_latent_dims}.\n")
    print("Start training the best model...")
    best_model = FullConnectionModel(best_latent_dims)
    x = np.concatenate([x_train, x_validation], axis=0)
    y = np.concatenate([y_train, y_validation], axis=0)
    bar = trange(epochs)
    loss_train = []
    accuracy_train = []
    loss_test = []
    accuracy_test = []
    W_train = []
    for epoch in bar:
        loss1, accuracy1 = trainOneStep(best_model, x, y, learning_rate)
        W_train.append(np.dot(best_model.W1, best_model.W2))
        loss_train.append(loss1)
        accuracy_train.append(accuracy1)
        loss2, accuracy2 = evaluate(best_model, mnist.test.images, mnist.test.labels)
        loss_test.append(loss2)
        accuracy_test.append(accuracy2)
        bar.set_description(f'Training the best model, epoch={epoch + 1: <3}, loss={loss1: <10.8}, accuracy={accuracy1: <8.6}')  # 给进度条加个描述
    bar.close()

    return best_model, loss_train, accuracy_train, loss_test, accuracy_test, W_train


# 评估模型
def evaluate(model, x, y):
    model.forward(x, y)
    loss = model.loss
    accuracy = computeAccuracy(model.h2_soft, y)
    return loss, accuracy

#可视化loss和accuracy曲线
def plot_loss(v_lo, t_lo):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()  # 共享x轴

    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("validation-loss")
    par1.set_ylabel("test-loss")

    # plot curves
    p1, = host.plot(range(len(v_lo)), v_lo, label="validation-loss")
    p2, = par1.plot(range(len(t_lo)), t_lo, label="test_loss")

    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.draw()
    plt.show()


def plot_acc(v_acc, t_acc):
    host = host_subplot(111)  # row=1 col=1 first pic
    plt.subplots_adjust(right=0.8)  # ajust the right boundary of the plot window
    par1 = host.twinx()  # 共享x轴

    # set labels
    host.set_xlabel("steps")
    host.set_ylabel("validation-accuracy")
    par1.set_ylabel("test-accuracy")

    # plot curves
    p1, = host.plot(range(len(v_acc)), v_acc, label="validation-accuracy")
    p2, = par1.plot(range(len(t_acc)), t_acc, label="test_accuracy")

    host.legend(loc=5)

    # set label color
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())

    plt.draw()
    plt.show()



if __name__ == '__main__':
    mnist = input_data.read_data_sets("dataset/", one_hot=True)

    model, loss_train, accuracy_train, loss_test, accuracy_test, W_train = train(mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels)
    i = 0
    imgs = list()
    for key in W_train:
        i += 1
        img = np.zeros((28, 28, 3)).tolist()
        pca = PCA(n_components=3)
        key1 = pca.fit_transform(key)
        img_min = key1[0][0]
        img_max = key1[0][0]
        for j in range(28):
            for k in range(28):
                for r in range(3):
                    m = 28 * 28 * r + (j + 1) * (k + 1)
                    img[j][k][r] = key[m // 10][m - 10 * (m // 10)]
                    if img_min > img[j][k][r]:
                        img_min = img[j][k][r]
                    if img_max < img[j][k][r]:
                        img_max = img[j][k][r]
        for j in range(28):
            for k in range(28):
                for r in range(3):
                    img[j][k][r] = int((img[j][k][r] - img_min) / (img_max - img_min) * 255)
        imgs.append(img)
        cv2.imwrite('%s.jpg' % i, np.array(img))
    with open("imgs.json", "w", encoding='utf-8') as f:
        f.write(json.dumps(imgs, ensure_ascii=False, indent=4, separators=(',', ':')))
    plot_loss(loss_train, loss_test)
    plot_acc(accuracy_train, accuracy_test)

