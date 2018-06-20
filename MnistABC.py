# -*- coding:utf-8 -*-

import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 引入mnist数据集

x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b) # 将evidence输入softmax函数

# 设置训练过程
y_ = tf.placeholder("float", [None,10]) # 为交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 利用反向传播算法，目标是最小化交叉熵，0.01为梯度下降算法的速率

#开始训练
init = tf.global_variables_initializer() # 初始化创建变量

sess = tf.Session() # 在Session中启动
sess.run(init)

# 开始循环训练1000次
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评估
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
