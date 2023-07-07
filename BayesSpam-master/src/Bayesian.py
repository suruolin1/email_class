from spam.spamEmail import spamEmailBayes
import re

#spam类对象
spam=spamEmailBayes()
#保存词频的词典
ggspamDict={}
zpspamDict={}
dyspamDict={}
normDict={}
testDict={}
#保存每封邮件中出现的词
wordsList=[]
wordsDict={}
#保存预测结果,key为文件名，值为预测类别
testResult={}
#分别获得正常邮件、垃圾邮件及测试文件名称列表
normFileList=spam.get_File_List(r"E:\email classification\BayesSpam-master\data\normal")
ggspamFileList=spam.get_File_List(r"E:\email classification\BayesSpam-master\data\ggspam")
zpspamFileList=spam.get_File_List(r"E:\email classification\BayesSpam-master\data\zpspam")
dyspamFileList=spam.get_File_List(r"E:\email classification\BayesSpam-master\data\dyspam")
testFileList=spam.get_File_List(r"E:\email classification\BayesSpam-master\data\test")
#获取训练集中正常邮件与垃圾邮件的数量
normFilelen=len(normFileList)
ggspamFilelen=len(ggspamFileList)
zpspamFilelen=len(zpspamFileList)
dyspamFilelen=len(dyspamFileList)
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

#获得广告类垃圾邮件中的词频
wordsDict.clear()
for fileName in ggspamFileList:
    wordsList.clear()
    for line in open("../data/ggspam/"+fileName):
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        spam.get_word_list(line,wordsList,stopList)
    spam.addToDict(wordsList, wordsDict)
ggspamDict=wordsDict.copy()

#获得诈骗类垃圾邮件中的词频
wordsDict.clear()
for fileName in zpspamFileList:
    wordsList.clear()
    for line in open("../data/zpspam/"+fileName):
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        spam.get_word_list(line,wordsList,stopList)
    spam.addToDict(wordsList, wordsDict)
zpspamDict=wordsDict.copy()

#获得钓鱼类垃圾邮件中的词频
wordsDict.clear()
for fileName in dyspamFileList:
    wordsList.clear()
    for line in open("../data/dyspam/"+fileName):
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        spam.get_word_list(line,wordsList,stopList)
    spam.addToDict(wordsList, wordsDict)
dyspamDict=wordsDict.copy()

# 测试邮件
for fileName in testFileList:
    testDict.clear( )
    wordsDict.clear()
    wordsList.clear()
    for line in open("../data/test/"+fileName):
        rule=re.compile(r"[^\u4e00-\u9fa5]")
        line=rule.sub("",line)
        spam.get_word_list(line,wordsList,stopList)
    spam.addToDict(wordsList, wordsDict)
    testDict=wordsDict.copy()
    #通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    wordProbList=spam.getTestWords(testDict, ggspamDict,normDict,normFilelen,ggspamFilelen)
    #对每封邮件得到的15个词计算贝叶斯概率  
    p1=spam.calBayes(wordProbList, ggspamDict, normDict)
    wordProbList=spam.getTestWords(testDict, zpspamDict,normDict,normFilelen,zpspamFilelen)
    p2=spam.calBayes(wordProbList, zpspamDict, normDict)
    wordProbList=spam.getTestWords(testDict, dyspamDict,normDict,normFilelen,dyspamFilelen)
    p3=spam.calBayes(wordProbList, dyspamDict, normDict)
    if(p1<0.9 and p2<0.9 and p3<0.9):
        testResult.setdefault(fileName,0)
    else:
        wordProbList = spam.getTestWords(testDict, zpspamDict, ggspamDict,ggspamFilelen, zpspamFilelen)
        p4 = spam.calBayes(wordProbList, zpspamDict, ggspamDict)
        wordProbList=spam.getTestWords(testDict, zpspamDict,dyspamDict,dyspamFilelen,zpspamFilelen)
        p5=spam.calBayes(wordProbList, zpspamDict, dyspamDict)
        wordProbList = spam.getTestWords(testDict, dyspamDict, ggspamDict,ggspamFilelen, dyspamFilelen)
        p6 = spam.calBayes(wordProbList, dyspamDict, ggspamDict)
        if (p4<0.9 and p6<0.9):#广告
            testResult.setdefault(fileName,1)
        elif(p4>0.9 and p5>0.9):#诈骗
            testResult.setdefault(fileName, 2)
        elif(p5<0.9 and p6>0.9):#钓鱼
            testResult.setdefault(fileName, 3)

#计算分类准确率（测试集中文件名低于1000的为正常邮件）
testAccuracy=spam.calAccuracy(testResult)
for i,ic in testResult.items():
    print(i+"/"+str(ic))
print(testAccuracy)  