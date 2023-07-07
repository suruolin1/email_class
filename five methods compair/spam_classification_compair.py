import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import string

from sklearn.preprocessing import LabelEncoder

PUNCT_TO_REMOVE = string.punctuation

from nltk import word_tokenize
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV  # 搜索适合的参数

# 读邮件数据CSV
train_email = pd.read_csv("data/train.csv", usecols=[2], encoding='utf-8')
train_label = pd.read_csv("data/train.csv", usecols=[1], encoding='utf-8')

# 数据预处理
def text_processing(text):
    text = text.lower()
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text)
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    # str.maketrans('', '', PUNCT_TO_REMOVE) 用于创建一个字符映射表，
    # 将文本中的特定字符映射为空字符。
    # text.translate() 方法使用该字符映射表，将文本中的特定字符替换为空字符。
    # PUNCT_TO_REMOVE 是一个字符串，包含了需要从文本中删除的标点符号。
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    # [stemmer.stem(word) for word in text.split()]
    # 使用词干提取器（stemmer）对每个单词进行词干提取，并生成一个新的列表。
    # 例如，如果text的值为"running dogs are fast",
    # 这行代码将把它转换为"run dog are fast"，其中每个单词都经过了词干提取。
    return text

# 将 text_processing 函数应用到 train_email['Email'] 列的每个元素上,即每一封邮件。
train_email['Email'] = train_email['Email'].apply(text_processing)

# 将内容转为list类型

# np.array(train_email)将train_email转换为NumPy数组。
# .reshape((1, len(train_email)))将数组的形状调整为(1, len(train_email))，
# 即一个行向量。[0]从调整后的数组中获取第一个元素，即行向量。
# .tolist()将行向量转换为Python列表。
train_email = np.array(train_email).reshape((1, len(train_email)))[0].tolist()
train_label = np.array(train_label).reshape((1, len(train_email)))[0].tolist()

# 构造训练集和验证集，取数据集中的80%作为训练集
train_num = int(len(train_email) * 0.8)
data_train = train_email[:train_num]
data_dev = train_email[train_num:]
label_train = train_label[:train_num]
label_dev = train_label[train_num:]

# # 使用词袋模型
vectorizer = CountVectorizer()
# CountVectorizer类会把文本全部转换为小写，然后将文本词块化。主要是分词，分标点
data_train_cnt = vectorizer.fit_transform(data_train)
data_test_cnt = vectorizer.transform(data_dev)

#变成TF-IDF矩阵，加入TF-IDF特征
transformer = TfidfTransformer()
data_train_tfidf = transformer.fit_transform(data_train_cnt)
data_test_tfidf = transformer.transform(data_test_cnt)

# 利用贝叶斯的方法
clf = MultinomialNB()
clf.fit(data_train_cnt, label_train)
score = clf.score(data_test_cnt, label_dev)
print("MultinomialNB score: ", score)
clf.fit(data_train_tfidf, label_train)
score = clf.score(data_test_tfidf, label_dev)
print("MultinomialNB tfidf score: ", score)

# 利用SVM的方法
svm = LinearSVC()
svm.fit(data_train_cnt, label_train)
score = svm.score(data_test_cnt, label_dev)
print("SVM score: ", score)
svm.fit(data_train_tfidf, label_train)
score = svm.score(data_test_tfidf, label_dev)
print("SVM score: ", score)

# 利用逻辑回归的方法
# max_iter=150 参数表示最大迭代次数为150次。
# penalty='l2' 参数表示使用L2正则化。
# solver='lbfgs' 参数表示使用LBFGS优化算法进行求解。
# random_state=0 参数表示随机数生成器的种子，以确保结果的可重复性，这里好像没什么用。
# lbfgs拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
lr_crf = LogisticRegression(max_iter=150,
                            penalty='l2', solver='lbfgs', random_state=0)
lr_crf.fit(data_train_tfidf, label_train)
score = lr_crf.score(data_test_tfidf, label_dev)
print("LR score: ", score)

# 利用随机森林的方法
# n_estimators=100 参数表示森林中树木的数量为100。
# max_depth=None 参数表示树木的最大深度不限制。
# verbose=0 参数表示不输出训练过程的冗长信息。
# n_jobs=-1 参数表示使用所有可用的处理器进行并行处理。
rf = RandomForestClassifier(random_state=0, n_estimators=100,
                            max_depth=None, verbose=0, n_jobs=-1)
rf.fit(data_train_tfidf, label_train)
score = rf.score(data_test_tfidf, label_dev)
print("RF score: ", score)

# 利用LightGBM的方法
lgb_clf = lgb.LGBMClassifier()
# lgb_clf.fit(data_train_tfidf, label_train)
# 使用网格搜索得到适当参数
param_test = {
    'max_depth': range(2, 3)
}
# 参数网格param_test，评分指标为ROC-AUC，交叉验证的折数为5。
gsearch = GridSearchCV(estimator=lgb_clf,
                       param_grid=param_test, scoring='roc_auc', cv=5)
gsearch.fit(data_train_tfidf, label_train)
score = gsearch.score(data_test_tfidf, label_dev)
print("LGBM score: ", score)

# 预测结果
result_lgbm = gsearch.predict(data_test_tfidf)
result_rf = rf.predict(data_test_tfidf)
result_lr = lr_crf.predict(data_test_tfidf)
result_svm = svm.predict(data_test_cnt)
result_nb = clf.predict(data_test_cnt)
print("NB confusion: ", confusion_matrix(label_dev, result_nb))
print("SVM confusion: ", confusion_matrix(label_dev, result_svm))
print("LR confusion: ", confusion_matrix(label_dev, result_lr))
print("RF confusion: ", confusion_matrix(label_dev, result_rf))
print("LGBM confusion: ", confusion_matrix(label_dev, result_lgbm))
