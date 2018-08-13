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

conn = pymysql.Connect(host='192.168.101.74',port=3306,user='root',passwd='123456',db='stockcn',charset='utf8')
cursor = conn.cursor()
sqlcount = "select name from hot where context like '%运输业概念%' "
answer = pd.read_sql(sql=sqlcount, con=conn)
amgdp = "select gongbuzhi from macro_economy where country = 'usa' and zhibiao like '%gdp年化季率终值%' and shijian > '2018-01-01'"
answer1 = pd.read_sql(sql=amgdp, con=conn)
yinyeehuaxia = "select REVENUE from income_bank where TICKER_SYMBOL = '600015' and END_DATE='2018-03-31' "
answer2 = pd.read_sql(sql=yinyeehuaxia, con=conn)
why = [
    "这个问题比较复杂，让我来给你慢慢解释，首先，本宝每天大量抓取热点及时新闻，今天有1000条，你要听吗？"
    “不要”
    "其次，爷每周还更新一次所有股票的背景资料，你想听谁的背景资料吗？本宝可以很有耐心的读给你听哦。"
    “不要”
    "最后就是上帝给我创造了算法，让我从众多的热点新闻中挑选最匹配的股票然后推荐给您"
]
conversation=[
    "给我推荐股票？",
     answer,
    "为什么？",
      why,
    "美国近期的gdp是多少？",
     answer1,
    "华夏银行2018年一季度的营业额是多少？",
     answer2,
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
