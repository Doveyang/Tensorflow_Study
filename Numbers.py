# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from os import listdir


def img2vector(filename):
    returnVector = np.zeros((1, 1024))
    fr = open(filename)
    for k in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVector[0, 32 * k + j] = int(lineStr[j])
    return returnVector


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def generate_Matrix(dir):
    File_List = listdir(dir)
    m = len(File_List)
    x_data = np.zeros((m, 1024))
    y_data = np.zeros((m, 1))
    for i in range(m):
        fileNameStr = File_List[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        y_data[i][0] = classNumStr
        x_data[i, :] = img2vector(dir + fileNameStr)
    return x_data, y_data


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


x_data, y_data = generate_Matrix('digits/trainingDigits/')
xs = tf.placeholder(tf.float32, [None, 1024])
ys = tf.placeholder(tf.float32, [None, 1])

prediction = add_layer(xs, 1024, 1, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # 未做batch

v_xs, v_ys = generate_Matrix('digits/testDigits/')
y_pre = sess.run(prediction, feed_dict={xs: v_xs})
print y_pre  # 输出全为1，而compute_accuracy却显示100%，非常奇怪。。。
print compute_accuracy(v_xs, v_ys)
