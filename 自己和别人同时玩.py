import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn  import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math
import keras
import pymysql

# hidden layer
rnn_unit = 128
# feature
input_size = 1
output_size = 1
lr = 0.0006
conn = pymysql.Connect(host='192.168.101.74',port=3306,user='root',passwd='123456',db='stockcn',charset='utf8')
cursor = conn.cursor()
sqlx = "select close_price,good_awareness,acc_or_rej  from day_price_alpha where symbol = 600004 order by price_date desc limit 100 "
sqlshindex = "select close_price  from shindex  order by datep desc limit 100"
dfx = pd.read_sql(sql=sqlx, con=conn)
x_data = preprocessing.minmax_scale(dfx, feature_range=(-1, 1))
dfy = pd.read_sql(sql=sqlshindex, con=conn)
y_data = preprocessing.minmax_scale(dfy, feature_range=(-1, 1))
def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans

xs = tf.placeholder(tf.float32,[None, input_size])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32,[None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

def emotion():
    close1 = addLayer(xs, input_size, 1, activity_function=tf.nn.tanh) # tanh是激励函数的一种
    costd1 = tf.layers.dropout(close1, rate=0.1, training=tf_is_training)
    close2 = addLayer(close1, 1, 1, activity_function=None)
    costd2 = tf.layers.dropout(close2, rate=0.1, training=tf_is_training)   # drop out 50% of inputs
    return costd2

# losscompare = tf.reduce_mean(tf.reduce_sum(tf.square(ys - compare()), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
# traincompare = tf.train.GradientDescentOptimizer(lr).minimize(losscompare)  # 选择梯度下降法

d_outemotion = tf.layers.dense(emotion(), 1)

lossemotion = tf.reduce_mean(tf.reduce_sum(tf.square(ys - d_outemotion), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
trainemotion = tf.train.GradientDescentOptimizer(lr).minimize(lossemotion)  # 选择梯度下降法

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
File = open("./data/emotion.txt", "w", encoding=u'utf-8', errors='ignore')

for k in range(1,50):
     sqlx = "select close_price,good_awareness  from day_price_alpha,symbol where day_price_alpha.symbol=symbol.symbol and symbol.id="
     sqlx += str(k)
     sqlx += " ORDER BY day_price_alpha.price_date desc  limit 100"
     sqlshindex = "select close_price  from shindex  order by datep desc limit 100"
     sqlsymbol = "select symbol  from symbol where symbol.id=  "
     sqlsymbol += str(k)
     fx = pd.read_sql(sql=sqlsymbol, con=conn)
     dfx = pd.read_sql(sql=sqlx, con=conn)
     x_data = preprocessing.minmax_scale(dfx, feature_range=(-1, 1))
     dfy = pd.read_sql(sql=sqlshindex, con=conn)
     y_data = preprocessing.minmax_scale(dfy, feature_range=(-1, 1))
     #emotion()
     for i in range(10000):
           loss_emotion = sess.run([lossemotion, trainemotion], feed_dict={xs: x_data[:, 0:1], ys: x_data[:, 1:2],tf_is_training: True})
           loss_compare = sess.run([lossemotion, trainemotion], feed_dict={xs: x_data[:, 0:1], ys: y_data,tf_is_training: True})
           if i % 10000 == 0:
               File.write(str(fx))
               File.write(str(loss_emotion))
               File.write(str(loss_compare) + "\n")


with tf.name_scope('loss'):
    lossemotion = tf.reduce_mean(tf.reduce_sum(tf.square((ys - d_outemotion)), reduction_indices=[1]))
    losscompare = tf.reduce_mean(tf.reduce_sum(tf.square((ys - d_outemotion)), reduction_indices=[1]))
    tf.summary.scalar('lossemotion', lossemotion)
    tf.summary.scalar('losscomparen', losscompare)

with tf.name_scope('train'):
        train_stepemotion = tf.train.GradientDescentOptimizer(lr).minimize(lossemotion)
        train_stepcmpare = tf.train.GradientDescentOptimizer(lr).minimize(losscompare)

sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train_stepemotion, feed_dict={xs: x_data[:, 0:1], ys: x_data[:, 1:2],tf_is_training: True})
    sess.run(train_stepcmpare, feed_dict={xs: x_data[:, 0:1], ys: y_data,tf_is_training: True})
    if i % 50 == 0:
        result1 = sess.run(merged, feed_dict={xs: x_data[:, 0:1], ys: x_data[:, 1:2],tf_is_training: True})
        result2 = sess.run(merged, feed_dict={xs: x_data[:, 0:1], ys: y_data,tf_is_training: True})
        writer.add_summary(result1, i)
        writer.add_summary(result2, i)


# if int((tf.__version__).split('.')[1]) < 12 and int(
#             (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
#             writer = tf.train.SummaryWriter('logs/', sess.graph)
# else:  # tensorflow version >= 0.12
#             writer = tf.summary.FileWriter("logs/", sess.graph)
#tensorboard --logdir=logs