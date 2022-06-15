import math
from tqdm import tqdm
from Datasets import DataLoader
from optimizers import Fixed, Adam
from LeNet import LeNet5
from Loss import CategoricalCrossEntropy
from Utils import one_hot, computeAccuracy, evaluate
import tensorflow as tf


# 训练模型和寻优
def train(x_train, y_train, x_validation, y_validation):
    epochs = 2
    learning_rate = 0.0001
    batch_size = 64

    print("Start training...\n")
    # 模型
    model = LeNet5(10)
    # 损失函数
    criterion = CategoricalCrossEntropy()
    # 优化器
    optimizer = Adam(model, learning_rate)

    total_size = math.ceil(y_train.shape[0] / batch_size)
    for epoch in range(epochs):
        data_loader = DataLoader(x_train, y_train, batch_size, True)
        # each batch
        with tqdm(total=total_size) as _tqdm:  # 使用需要的参数对tqdm进行初始化
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, epochs))  # 设置前缀 一般为epoch的信息
            for x, y in data_loader():
                pre_y = model.forward(x)
                # 计算误差
                loss = criterion(y, pre_y)
                # 后向传播
                model.backwards(criterion.gradient)
                # 调整参数
                optimizer.step()
                # acc
                acc = computeAccuracy(pre_y, y)
                # 记录acc和loss用于绘图
                # accuracy = evaluate(model, x_validation, y_validation)
                model.train_acc.append(acc)
                # model.test_acc.append(accuracy)
                model.loss.append(loss)
                _tqdm.set_postfix(
                    {"loss": '{:.6f}'.format(loss), "acc": '{:.6f}'.format(acc)})  # 输入一个字典
                _tqdm.update(1)  # 设置你每一次想让进度条更新的iteration 大小
        model.eval()
        validation_accuracy = evaluate(model, x_validation, y_validation)
        model.train()
        print(f"validation_accuracy={validation_accuracy}.\n")
        model.log_visualization()
    return model


# Minst
def hand_write():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train = x_train / x_train.mean()
    X_test = x_test / x_test.mean()
    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    model = train(X_train, y_train, X_test, y_test)
    accuracy = evaluate(model, X_test, y_test)
    print(f'Evaluate the best model, test accuracy={accuracy:0<8.6}.')
    path = "Models/LeNet_Minst1.John"
    model.save(path)
    print("save models at:", path)


if __name__ == '__main__':
    hand_write()
