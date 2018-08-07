#!/usr/bin/python
from wxpy import *
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import pymysql
import pandas as pd
bot = Bot(cache_path=True)
my_bot = ChatBot("Training demo")
#chatbot=ChatBot("wwjtest",read_only=True) //否则bot会学习每个输入
my_bot.set_trainer(ListTrainer)

conn = pymysql.Connect(host='127.0.0.1',port=3306,user='root',passwd='',db='stockcn',charset='utf8')
cursor = conn.cursor()
sqlcount = "select name from hot where context like '%运输业概念%' "
answer = pd.read_sql(sql=sqlcount, con=conn)
conversation=[
    "给我推荐股票？",
     answer,
]
my_bot.train(conversation)
myself = bot.friends().search('forex')[0]

# 向文件传输助手发送消息
myself.send('Hello!')


@bot.register(myself)
def reply_my_friend(msg):
       print(msg)
       print(answer)
       ans = my_bot.get_response(msg)
       myself.send(ans)
       return myself.get_response(msg.text).tex# 使用机器人进行自动回复
# 堵塞线程，并进入 Python 命令行
embed()