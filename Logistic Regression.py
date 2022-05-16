# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:43:48 2021

@author: yuwen
"""

#### Word Bag
import nltk
from nltk.corpus import wordnet
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
defaultStopwords = stopwords.words('english')

dum_train = open('d_train', 'r')
snp_train = open('s_train', 'r')

trainPos = []
dum_line = dum_train.readline()
while dum_line:
    trainPos.append(dum_line)
    dum_line = dum_train.readline()
    
    
trainNeg = []
snp_line = snp_train.readline()
while snp_line:
    trainNeg.append(snp_line)
    snp_line = snp_train.readline()
    
dum_train.close()
snp_train.close()


from nltk.tokenize import RegexpTokenizer
newTrainPos = []
for s in trainPos:
    new = ''
    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
    noPunct = tokenizer.tokenize( s )
    for word in noPunct:
        if word.lower() not in defaultStopwords:
            new += word.lower()
            new += ' '
    newTrainPos.append(new)
#print(newTrainPos)
#print()

newTrainNeg = []
for s in trainNeg:
    new = ''
    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
    noPunct = tokenizer.tokenize( s )
    for word in noPunct:
        if word.lower() not in defaultStopwords:
            new += word.lower()
            new += ' '
    newTrainNeg.append(new)
#print(newTrainNeg)


V = set()
bagPos = []   # bagPos: words in "positive" class
for pos in newTrainPos:
    tokenList = nltk.word_tokenize(pos)
    for w in tokenList:
        bagPos.append(w)
        V.add(w)


bagNeg = []   # bagNeg: words in "negative" class
for neg in newTrainNeg:
    tokenList = nltk.word_tokenize(neg)
    for w in tokenList:
        bagNeg.append(w)
        V.add(w)
        
        
        
        
        
######## LR Model test
import nltk
from nltk.corpus import wordnet
import math
#
pos_word = set(bagPos)
neg_word = set(bagNeg)
#
def zValue(fv,w,b):
    return b + sum([fv[i]*w[i] for i in range(len(fv))])
def sigmoid(N):
    return 1/ ( 1 + math.exp(-N))
#


import csv
d_result = open ('./Dumbledore_LR_Result.csv', 'w', newline = '')
s_result = open ('./Snape_LR_Result.csv', 'w', newline = '')
headList = ['content', 'p1', 'whether the character']
writer1 = csv.DictWriter(d_result, fieldnames = headList)
writer2 = csv.DictWriter(s_result, fieldnames = headList)
writer1.writeheader()
writer2.writeheader()



print('TEST START')
d_test = open('d_test', 'r')
s_test = open('s_test','r')
testPos = []
testNeg = []

d_line = d_test.readline()
s_line = s_test.readline()

while d_line:
    testPos.append(d_line)
    d_line = d_test.readline()
while s_line:
    testNeg.append(s_line)
    s_line = s_test.readline()

d_test.close()
s_test.close()


print('DUMBLEDORE')
for fullString in testPos:
    print("-"*30)
    fv = [0]*2
    for w in nltk.word_tokenize(fullString):
        if w in pos_word:
            fv[0] += 1
        if w in neg_word:
            fv[1] += 1
    
    z = zValue(fv = fv,w = [3, -1], b = 0)
    p1 = sigmoid(z)
    print(f'Test string = "{fullString}"')
    print(f"fv = {fv}")
    print(f'z = {z}')
    print(f'p1 = {p1}')
    if p1> 0.5:
        print("Classified as postive")
        writer1.writerow({headList[0]:fullString, headList[1]:p1, headList[2]:'1'})
    else:
        print("Classified as negative")
        writer1.writerow({headList[0]:fullString, headList[1]:p1, headList[2]:'0'})
d_result.close()

print('\n\nSNAPE')
for fullString in testNeg:
    print("-"*30)
    fv = [0]*2
    for w in nltk.word_tokenize(fullString):
        if w in pos_word:
            fv[0] += 1
        if w in neg_word:
            fv[1] += 1
    
    z = zValue(fv = fv,w = [1, -2], b = 0)
    p1 = sigmoid(z)
    print(f'Test string = "{fullString}"')
    print(f"fv = {fv}")
    print(f'z = {z}')
    print(f'p1 = {p1}')
    if p1> 0.5:
        print("Classified as postive")
        writer2.writerow({headList[0]:fullString, headList[1]:p1, headList[2]:'0'})
    else:
        print("Classified as negative")
        writer2.writerow({headList[0]:fullString, headList[1]:p1, headList[2]:'1'})
s_result.close()
# Part 2
########  
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.sparse import csr_matrix
#
def prediction_measure(y_test, y_predicted):
    CM = confusion_matrix(y_test, y_predicted)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    #
    print(CM)
    print("Accuracy = ", accuracy_score(y_test, y_predicted))
    print("Precision = ", precision_score(y_test, y_predicted))
    print("Recall = ", recall_score(y_test, y_predicted))

posDocs = []   
negDocs = []  
docsToClassify = []  
fp = open('d_train', 'r') 
for line in fp:
    posDocs.append(line.strip())
fp = open('s_train', 'r') 
for line in fp:
    negDocs.append(line.strip())
theDocs = posDocs + negDocs ;
labelsPosDocs = [1 for i in posDocs]
labelsNegDocs = [0 for i in negDocs]
labels = labelsPosDocs + labelsNegDocs
# Original Approach
cv = CountVectorizer(binary=False,max_df=0.95)
cv.fit_transform(theDocs)
counts = cv.transform(theDocs)   
x_train, x_test, y_train, y_test  = train_test_split( counts, labels,  test_size=0.2, random_state=1)

# LR
print("-"*20)
print("LR Evaluation")
model_logisticReg = LogisticRegression(random_state=0,max_iter=10000000).fit(x_train,y_train)
y_predicted_LR = model_logisticReg.predict(x_test)
prediction_measure(y_test, y_predicted_LR)
