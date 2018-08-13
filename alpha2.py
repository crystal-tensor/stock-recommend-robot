import pandas as pd
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn  import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math

# hidden layer
rnn_unit = 128
# feature
input_size = 40
output_size = 1
lr = 0.0006
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

for i in range(12000):
    loss_dropout, d_train = sess.run([d_loss, d_train], feed_dict={xs: x_data, ys: y_data,tf_is_training: True})

    if i % 12000 == 0:
        base_path = saver.save(sess, "module/alpha.model")
        #print("loss_overfiting",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        #print("loss_dropout", sess.run(d_loss, feed_dict={xs: x_data, ys: y_data,tf_is_training: True}))
#         plt.cla()
#         plt.text(0.1, 0, 'loss_overfiting=%.04f' % loss_overfiting, fontdict={'size': 20, 'color': 'red'})
#         plt.text(0.1, 1, 'loss_dropout=%.04f' % loss_dropout, fontdict={'size': 20, 'color': 'green'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()

#
# if int((tf.__version__).split('.')[1]) < 12 and int(
#             (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
#             writer = tf.train.SummaryWriter('logs/', sess.graph)
# else:  # tensorflow version >= 0.12
#             writer = tf.summary.FileWriter("logs/", sess.graph)
#tensorboard --logdir=logs