
import thulac

#默认模式，分词的同时进行词性标注
test1 = thulac.thulac()
text1 = test1.cut("华夏银行2018年一季度的营业额是多少")
print(text1)


#只进行分词
test2 = thulac.thulac(seg_only=True)
text2 = test2.cut("美国近期的gdp是多少")
print(text2)

