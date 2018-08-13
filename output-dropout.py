import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn  import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math
import codecs


# hidden layer
rnn_unit = 128
# feature
input_size = 40
output_size = 1
lr = 0.0006
k=4
# csv_file = 'stock3005.csv'
csv_file = 'fof基金20170731-1031.csv'
f = open(csv_file, 'r', encoding=u'utf-8', errors='ignore')
df = pd.read_csv(f)
df.dropna(inplace=True)

def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans

x_data = preprocessing.minmax_scale(df.iloc[:, 3:43].values,feature_range=(-1,1))
y_data = preprocessing.minmax_scale(df.iloc[:, 43:44].values,feature_range=(-1,1))

x_data_output = df.iloc[:, 1:2].values
# File = open("ser.txt", "w", encoding=u'utf-8', errors='ignore')
# File.write(str(df.iloc[:, 1:3].values) + "\n")
# x_data = df.iloc[:, 3:56].values
# y_data = df.iloc[:, 56:57].values

xs = tf.placeholder(tf.float32, [None, input_size])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

l1 = addLayer(xs, input_size, 1, activity_function=tf.nn.tanh) # relu是激励函数的一种
d1 = tf.layers.dropout(l1, rate=0.1, training=tf_is_training)
l2 = addLayer(l1, 1, 1, activity_function=None)
d2 = tf.layers.dropout(l2, rate=0.1, training=tf_is_training)   # drop out 50% of inputs

loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - l2)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)  # 选择梯度下降法

d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(ys, d_out)
d_train = tf.train.AdamOptimizer(lr).minimize(d_loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
File = open("fund-score-12-alpha2-3.txt", "w",encoding=u'utf-8', errors='ignore')
with tf.Session() as sess:
    saver.restore(sess, "module/alpha520-12.model")
    for step in range(len(x_data) - 1):
        prob = sess.run(l2, feed_dict={xs: [x_data[step]]})
        test_y = np.array(y_data)
        File.write(str(x_data_output[step]))
        File.write(str(y_data[step]))
        File.write(str(prob) + "\n")
       # File.close()
     #acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
     #avg_diff = np.average(np.abs(y_data - test_y[:len(test_predict)]))
    # if test_predict()>0.5:
    #     accute=1
    # else:
    #     accute=0
   # print("avg_diff=", avg_diff, ", acc=", acc)