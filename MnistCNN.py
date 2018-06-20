# -*- coding:utf-8 -*-

import tensorflow as tf 
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)     #下载并加载mnist数据
x = tf.placeholder(tf.float32, [None, 784])                        #输入的数据占位符
y_ = tf.placeholder(tf.float32, shape=[None, 10])            #输入的标签占位符

# 为每个像素点生成W
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# 因为使用的是ReLu神经元，为了避免出现都是0的情况，所以初始化的时候使用较小的正数
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 2*2的卷积核
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') 
  # x: 卷积层输入 shape=[batch, height, width, channels]，batch是输入的图片数量
  # filter: 卷积核 shape=[height, width, in_channels, out_channels]， in_channels: 图片的深度 out_channels: 卷积核的数量
  # stride: 卷积步长 shape=[batch_stride, height_stride, width_stride, channels_stride]
  # padding: 控制卷积核处理边界的策略，same意味着填满边界

def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  '''
  池化输入，tensor的shape同input_image, 一般为卷积后图像feature_map
  ksize对应输入图像各维度的池化尺寸；
  第一个参数对应batch方向上的池化尺寸，
  第二、三参数对应图像高宽上的池化尺寸，
  第四个参数对应channels方向上的池化尺寸，一般不在batch和channel方向上做池化，因此通常为[1,height,width,1]
  strides与padding同卷积函数tf.nn.conv2d()
  '''

#构建网络

'''
第一层卷积，32个卷积核，卷积核为5*5，每张图片输出32个feature map
'''
x_image = tf.reshape(x, [-1,28,28,1])         # 转换输入数据shape,以便于用于网络中，-1代表将x变换成一维矩阵，28*28是长宽，1是channel（灰度图为1）
W_conv1 = weight_variable([5, 5, 1, 32])      # 32个卷积核，每个5*5， channel是1
b_conv1 = bias_variable([32])       		  # 32个卷积核对应32个偏移量
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     #第一个卷积层，并用ReLu函数进行处理，引入非线性
h_pool1 = max_pool(h_conv1)                                  #第一个池化层

'''
第二层卷积，64个卷积核，卷积核为5*5，每张图片输出64个feature map
'''
W_conv2 = weight_variable([5, 5, 32, 64])	  # 第二层64个卷积核，5*5， channel是32，因为上一层每张图片输出了32个feature map
b_conv2 = bias_variable([64])				  # 64个卷积核对应64个偏移量
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #第二个卷积层
h_pool2 = max_pool(h_conv2)                                   #第二个池化层
'''
由于池化的尺寸是2*2，虽然每次使用卷积核的过程中28*28会变成24*24，但是由于padding的存在，所以输入输出尺寸一样
但是池化会把尺寸变小，所以整个过程是：28x28->28x28->14x14->14x14->7x7.
'''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout层

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #softmax层


cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.Session() # 在Session中启动

sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print ("test accuracy %g"%accuracy.eval(session=sess, feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))