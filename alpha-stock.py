import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#定义常量
rnn_unit=12       #hidden layer units
input_size=53
output_size=1
lr=0.0006         #学习率
#——————————————————导入数据——————————————————————
f=open('stock20170331-0721englis.csv')
df=pd.read_csv(f)     #读入股票数据
data=df.iloc[:,3:57].values  #取第3-10列


#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=2152):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:53]
       y=normalized_train_data[i:i+time_step,53,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y



#获取测试集
def get_test_data(time_step=20,test_begin=2152):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[]
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:53]
       y=normalized_test_data[i*time_step:(i+1)*time_step,53]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:53]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,53]).tolist())
    return mean,std,test_x,test_y



#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm(batch_size=80,time_step=15,train_begin=0,train_end=2152):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])    #batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
   #module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        base_path = saver.save(sess, "module/alpha120.model")
        #saver.restore(sess, module_file)
        #重复训练10000次
        for i in range(120):
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            # print(i,loss_)
            if i % 10==0:
                print("保存模型：",saver.save(sess,base_path,global_step=i))


#train_lstm()


#————————————————预测模型————————————————————
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
  #  Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        #module_file = tf.train.latest_checkpoint()
        saver.restore(sess, "module/alpha500.model")
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[53]+mean[53]
        test_predict=np.array(test_predict)*std[53]+mean[53]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        avg_diff = np.average(np.abs(test_predict - test_y[:len(test_predict)]))
        # if test_predict()>0.5:
        #     accute=1
        # else:
        #     accute=0
        print("avg_diff=", avg_diff, ", acc=", acc)
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()


#train_lstm()
prediction()


# 我很抱歉给你做了2年多都最后还是亏了，算起来到今天我做外汇10年了，但是我发现我始终get不到外汇的点，
# 所以总是亏钱的时候多过赚钱的时候。我一直想我如果放弃了可能就真的亏损了，所以虽然一直亏损但我还是一直坚持，
# 我一直报持着乐观的态度，我想也许那天我突然觉悟了，或者我找到了一个好的方法，我就能开始赚钱了。但是，
# 到今天我觉得还是放弃可能更好，在过去那么多年中我能总结的都总结了，我能意识到的不好的习气我也都有意识的去改过，虽然人性
# 不容易改，但是我觉得我还是克服了我自身的一些不好习气，佛教讲的贪嗔痴慢疑，我都刻意断除过，虽然这不太可能完全断除，
# 但是我觉得我还是有改过，比如，再也不买彩票了，这是断了贪心，当然这只是外汇交易的在心理层面的东西，更重要的是我想还是
# 我个人没法搞明白影响外汇走势的几个重要因素，我的 意思是我个人没办法全面了解外汇的方方面面，可能我搞懂了一些，但是这些东西
# 在交易的当下变成次要的原因了，而更重要的事虽然我也知道但是我并没有在交易的当下把他作为重要的事来看待，往往过后才意识到其实也都晚了。
# 外汇变化实在太快，虽然没有有很多时间去研究，但其实我并不清楚当下什么才是重要的，我大概都了解到了，但是我不没有搞明白什么才是最重要的。
# 影响外汇的因素太多，
