import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# 定义回调函数，如果模型的准确率大于0.999则停止训练
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.9999):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()
# 加载手写字体识别数据集, 其中训练集为60000张28*28的灰度图像，测试集为10000张
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 卷积神经网络调整
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
# 归一化操作，将数值映射到0-1之前的数字，方便运算
train_images = train_images / 255.0
test_images = test_images / 255.0

# 模型的构建方面我们借助keras来完成，只需要讲需要的层叠加到一起，并指明我们所需的优化器和损失函数即可
# 卷积神经网络模型
model = tf.keras.models.Sequential([
    # 卷积层，该卷积层的输出为32个通道，卷积核的大小是3*3，激活函数为relu，设置输入维度
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # 添加池化层，池化的kernel大小是2*2
    tf.keras.layers.MaxPooling2D(2, 2),
    # Add another convolution
    # 卷积层，输出为64个通道，卷积核大小为3*3，激活函数为relu
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # 池化层，最大池化，对2*2的区域进行池化操作
    tf.keras.layers.MaxPooling2D(2, 2),
    # Now flatten the output. After this you'll just have the same DNN structure as the non convolutional version
    # 将二维的输出转化为一维
    tf.keras.layers.Flatten(),
    # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
    # 通过softmax函数将模型输出为类名长度的神经元上，激活函数采用softmax对应概率值
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
# 输出模型
model.summary()

# 模型训练
# 指明模型的训练参数，优化器为adam优化器，损失函数为交叉熵损失函数
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=25, verbose=1, validation_data=[test_images,test_labels], callbacks=[callbacks])
loss, acc = model.evaluate(test_images, test_labels)
print('模型的测试准确率为{}'.format(acc))


# 评估训练模型
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('fig.png')
plt.show()

# 保存模型
model.save('.cnet1.pkl')
print('卷积模型保存成功！')

# result = np.argmax(model.predict(test_images[2].reshape(1, 28, 28, 1)))
# print(result)
