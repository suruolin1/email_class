from tkinter import *
import tkinter as tk
from tkinter.messagebox import *#导入messagebox子模块
from spam.spamEmail import spamEmailBayes
import re

#spam类对象
spam=spamEmailBayes()
#保存词频的词典
spamDict={}
normDict={}
testDict={}
#保存每封邮件中出现的词
wordsList=[]
wordsDict={}
#保存预测结果,key为文件名，值为预测类别
testResult={}
result = ""
#分别获得正常邮件、垃圾邮件及测
# 试文件名称列表
normFileList=spam.get_File_List(r"E:\email classification\Bayesian model class project\data\normal")
spamFileList=spam.get_File_List(r"E:\email classification\Bayesian model class project\data\spam")
testFileList=spam.get_File_List(r"E:\email classification\Bayesian model class project\data\test")
#获取训练集中正常邮件与垃圾邮件的数量
normFilelen=len(normFileList)
spamFilelen=len(spamFileList)
#获得停用词表，用于对停用词过滤
stopList=spam.getStopWords()

#获得正常邮件中的词频
for fileName in normFileList:
    wordsList.clear()
    for line in open("../data/normal/"+fileName):
        #过滤掉非中文字符
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        #将每封邮件出现的词保存在wordsList中
        spam.get_word_list(line,wordsList,stopList)
    #统计每个词在所有邮件中出现的次数
    spam.addToDict(wordsList, wordsDict)
normDict=wordsDict.copy()  

#获得垃圾邮件中的词频
wordsDict.clear()
for fileName in spamFileList:
    wordsList.clear()
    for line in open("../data/spam/"+fileName):
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        spam.get_word_list(line,wordsList,stopList)
    spam.addToDict(wordsList, wordsDict)
spamDict=wordsDict.copy()

#界面设计部分
root=tk.Tk()
root.geometry("1200x700+200+200")#对应的格式为宽乘以高加上水平偏移量加上垂直偏移量
label=Label(root,text="请输入邮件的地址",font=("方正舒体",18))
label.pack(ipadx=120)#调用pack方法将label标签显示在主界面，后面也会用到就不一一解释了
data=StringVar()#创建可编数据data
entry =Entry(root,textvariable=data,font="Arial")#创建labal组件并将其与data关联
entry.pack(ipadx=100)
#按钮响应函数
def callback():
    testDict = {}
    fileName = entry.get()
    testDict.clear()
    wordsDict.clear()
    wordsList.clear()
    for line in open("../data/test/" + fileName):
        #识别所有的中文部分
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", line)
        spam.get_word_list(line, wordsList, stopList)
    #将邮件内容输入到文本框
    # 设置文本格式tag
    txt.tag_config('tag_1', background='yellow',foreground='red', font=("隶书", 18))
    for line in open("../data/test/" + fileName):
        txt.insert(END, line,'tag_1')
    #进行分析
    spam.addToDict(wordsList, wordsDict)
    testDict = wordsDict.copy()
    # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    wordProbList = spam.getTestWords(testDict, spamDict, normDict,normFilelen, spamFilelen)
    # 对每封邮件得到的15个词计算贝叶斯概率
    p = spam.calBayes(wordProbList, spamDict, normDict)
    #将每个词的条件概率输入到文本框
    # 设置文本格式tag
    txt2.tag_config('tag_2',background='LightPink',foreground='blue', font=("微软雅黑", 19))
    for i, ic in wordProbList.items():
        txt2.insert(END, i + "/" + str(ic) + "\n", 'tag_2')
    if (p > 0.9):
        testResult.setdefault(fileName, 1)
        showinfo("分类结果","所选邮件为垃圾邮件")
        print("垃圾邮件")
    else:
        testResult.setdefault(fileName, 0)
        showinfo("分类结果","所选邮件为正常邮件")
        print("正常邮件")

button = Button(root, text='---确定---', command=callback,background="lightgreen",font=("微软雅黑",15))
button.pack()
txt=Text(root, width=150, height=20)
scrollBar = Scrollbar(root)
scrollBar.pack(side=RIGHT, fill=Y)
scrollBar.config(command=txt.yview)
txt.pack()
label=Label(root,text="各词的条件概率",font=("方正舒体",18))
label.pack()

def create():
    top = tk.Toplevel()
    top.title('Python')
    top.geometry("600x600+200+200")
    label = Label(top, text="请输入文件夹的地址", font=("方正舒体", 18))
    label.pack()
    data = StringVar()
    entry = Entry(top, textvariable=data,font="Arial")
    entry.pack()

    def callback():
        testDict = {}
        testFileList = spam.get_File_List(r"E:\email classification\Bayesian model class project\data\test")
        for fileName1 in testFileList:
            testDict.clear()
            wordsDict.clear()
            wordsList.clear()
            for line in open("../data/test/" + fileName1):
                rule = re.compile(r"[^\u4e00-\u9fa5]")
                line = rule.sub("", line)
                spam.get_word_list(line, wordsList, stopList)
            Minitxt.tag_config('tag_1', background='lightgreen',foreground='red', font=("隶书", 18))
            for line in open("../data/test/" + fileName1):
                Minitxt.insert(END, line,'tag_1')
            spam.addToDict(wordsList, wordsDict)
            testDict = wordsDict.copy()
            # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
            wordProbList = spam.getTestWords(testDict, spamDict, normDict,
                                             normFilelen, spamFilelen)
            # 对每封邮件得到的15个词计算贝叶斯概率
            p = spam.calBayes(wordProbList, spamDict, normDict)
            if (p > 0.9):
                testResult.setdefault(fileName1, 1)
            else:
                testResult.setdefault(fileName1, 0)
    button = Button(top, text='---确定---', command=callback,background="lightblue",font=("微软雅黑",15))
    button.pack()

    def analyse():
        testAccuracy = spam.calAccuracy(testResult)
        Minitxt2.tag_config('tag_2', background='PaleVioletRed',foreground='blue', font=("微软雅黑", 19))
        for i, ic in testResult.items():
            Minitxt2.insert(END, i + "/" + str(ic) + "\n",'tag_2')
        showinfo("分类准确率", "分类准确率为：" + str(testAccuracy))
        print(testAccuracy)

    Minitxt = Text(top, width=150, height=20)
    Minitxt.pack()
    button2 = Button(top, text='---预测---', command=analyse,background="orange",font=("微软雅黑",15))
    button2.pack()
    Minitxt2 = Text(top, width=150, height=25)
    Minitxt2.pack()

txt2=Text(root, width=150, height=25)
scrollBar2 = Scrollbar(root)
scrollBar2.pack(side=RIGHT, fill=Y)
scrollBar2.config(command=txt2.yview)
txt2.config(yscrollcommand=scrollBar.set)
txt2.pack()
button2 = Button(root, text='批量查看准确率', command=create,background="lightyellow",font=("微软雅黑",15)).pack()

mainloop()