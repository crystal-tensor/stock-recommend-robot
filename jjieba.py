import jieba

# 全模式
test1 = jieba.cut("美国最近的gdp是多少", cut_all=True)
print("全模式: " + "| ".join(test1))

# 精确模式
test2 = jieba.cut("华夏银行2018年一季度的营业额是多少？", cut_all=False)
print("精确模式: " + "| ".join(test2))

# 搜索引擎模式
test3 = jieba.cut_for_search("意大利的cpi是多少？")
print("搜索引擎模式:" + "| ".join(test3))

# {
#     "text": "美国最近的gdp是多少",
#     "intent": "宏观经济数据",
#     "entities": [
#         {
#             "country": 美国,
#             "时间": 近期,
#             "指标": "gpd",
#             "value": "是多少？"
#         }
#     ]
# },
#
# {
#     "text": "华夏银行2018年一季度的营业额是多少",
#     "intent": "财务数据",
#     "entities": [
#         {
#             "company": 华夏银行,
#             "时间": 2018年一季度,
#         "指标": "营业额",
# "value": "是多少？"
# }
# ]
# }
#
# {
#     "text": "阿里巴巴的股价是多少",
#     "intent": "行情数据",
#     "entities": [
#         {
#             "company": 阿里巴巴,
#             "时间": 今天，离现在最近一天的,
#                      "指标": "股价",
# "value": "是多少？"
# }
# ]
