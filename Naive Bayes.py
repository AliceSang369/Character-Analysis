# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 08:47:46 2021

@author: yuwen
"""

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

def count_W_inList(w, aList):
	count = 0
	for w2 in aList:
		if w2 == w:
			count += 1
	return(count)

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


total = len(trainPos) + len(trainNeg)
ProbPos = len(trainPos) / total
ProbNeg  = len(trainNeg) / total
#print(total, ProbPos, ProbNeg)

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
                
#---------------------------TEST-------------------------
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


newTestPos = []
for s in testPos:
    new = ''
    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
    noPunct = tokenizer.tokenize( s )
    for word in noPunct:
        if word.lower() not in defaultStopwords:
            new += word.lower()
            new += ' '
    newTestPos.append(new)
print(newTestPos)    
    
newTestNeg = []
for s in testNeg:
    new = ''
    tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator
    noPunct = tokenizer.tokenize( s )
    for word in noPunct:
        if word.lower() not in defaultStopwords:
            new += word.lower()
            new += ' '
    newTestNeg.append(new)
print(newTestNeg)


#Excel output
import csv
d_result = open ('./Dumbledore_NB_Result.csv', 'w', newline = '')
s_result = open ('./Snape_NB_Result.csv', 'w', newline = '')
headList = ['content', 'prob_pos', 'prob_neg', 'whether the character']
writer1 = csv.DictWriter(d_result, fieldnames = headList)
writer2 = csv.DictWriter(s_result, fieldnames = headList)
writer1.writeheader()
writer2.writeheader()

### Dumbledore (Pos)
for posTest in newTestPos:
    posTestList = nltk.word_tokenize(posTest)
    test = [ w for w in posTestList if w in V]     
    print("\nTest query = ")
    print(test)

    likelihoodW_pos = {} # dictionary of probabilities for positive class

    # create the denominator for the positive class
    # for each word in the complete vocabulary, sum up (count(w,c) + 1)
    denominator = 0
    for w in V:
        denominator += ( count_W_inList(w,bagPos) + 1)
        #print("debug: denominator postive class = " + str(denominator))

    # for each query word w, get the likelihood[w,postive]
    for w in test:
        likelihoodW_pos[w] =  (count_W_inList(w,bagPos) + 1) / denominator
        #print("debug: numerator pos class for w = " +  w + "  = " + str(count_W_inList(w,bagPos) + 1))

    print("likelihoodW_pos = ")
    print(likelihoodW_pos)


    likelihoodW_neg = {} # dictionary of probabilities for negative class

    # create the denominator for the negative class
    # for each word in the complete vocabulary, sum up (count(w,c) + 1)
    denominator = 0
    for w in V:
        denominator += ( count_W_inList(w,bagNeg) + 1)
        #print("debug: denominator negative class = " + str(denominator))

    # for each query word w, get the likelihood[w,negative]
    for w in test:
        likelihoodW_neg[w] =  (count_W_inList(w,bagNeg) + 1) / denominator
        #print("debug: numerator negative class for w = " +  w + "  = " + str(count_W_inList(w,bagNeg) + 1))

    print("likelihoodW_neg = ")
    print(likelihoodW_neg)

    # final calculations

    # P(-) P(S | -) 
    # where S is the test query Sentence, S = "predictable with no fun"
    finalProbNeg = ProbNeg 
    for w in test:
        finalProbNeg *=  likelihoodW_neg[w]
    print("Prob negative = " + str(finalProbNeg) )

    # P(+) P(S | +) 
    # where S is the test query Sentence, S = "predictable with no fun"
    finalProbPos = ProbPos 
    for w in test:
        finalProbPos *=  likelihoodW_pos[w]
    print("Prob positive = " + str(finalProbPos) )


    if (finalProbPos > finalProbNeg):
        print("\nModel predicts the test query belongs in the POSITVE class")
        writer1.writerow({headList[0]:posTest, headList[1]:str(finalProbNeg), headList[2]:str(finalProbPos), headList[3]:'1'})
    else:
        print("\nModel predicts the test query belongs in the NEGATIVE class")
        writer1.writerow({headList[0]:posTest, headList[1]:str(finalProbNeg), headList[2]:str(finalProbPos), headList[3]:'0'})
    

d_result.close()
    
    
#### Snape (Neg)
for negTest in newTestNeg:
    negTestList = nltk.word_tokenize(negTest)
    test = [ w for w in negTestList if w in V]     
    print("\nTest query = ")
    print(test)

    likelihoodW_pos = {} # dictionary of probabilities for positive class

    # create the denominator for the positive class
    # for each word in the complete vocabulary, sum up (count(w,c) + 1)
    denominator = 0
    for w in V:
        denominator += ( count_W_inList(w,bagPos) + 1)
        #print("debug: denominator postive class = " + str(denominator))

    # for each query word w, get the likelihood[w,postive]
    for w in test:
        likelihoodW_pos[w] =  (count_W_inList(w,bagPos) + 1) / denominator
        #print("debug: numerator pos class for w = " +  w + "  = " + str(count_W_inList(w,bagPos) + 1))

    print("likelihoodW_pos = ")
    print(likelihoodW_pos)


    likelihoodW_neg = {} # dictionary of probabilities for negative class

    # create the denominator for the negative class
    # for each word in the complete vocabulary, sum up (count(w,c) + 1)
    denominator = 0
    for w in V:
        denominator += ( count_W_inList(w,bagNeg) + 1)
        #print("debug: denominator negative class = " + str(denominator))

    # for each query word w, get the likelihood[w,negative]
    for w in test:
        likelihoodW_neg[w] =  (count_W_inList(w,bagNeg) + 1) / denominator
        #print("debug: numerator negative class for w = " +  w + "  = " + str(count_W_inList(w,bagNeg) + 1))

    print("likelihoodW_neg = ")
    print(likelihoodW_neg)


    # final calculations

    # P(-) P(S | -) 
    # where S is the test query Sentence, S = "predictable with no fun"
    finalProbNeg = ProbNeg 
    for w in test:
        finalProbNeg *=  likelihoodW_neg[w]
    print("Prob negative = " + str(finalProbNeg) )

    # P(+) P(S | +) 
    # where S is the test query Sentence, S = "predictable with no fun"
    finalProbPos = ProbPos 
    for w in test:
        finalProbPos *=  likelihoodW_pos[w]
    print("Prob positive = " + str(finalProbPos) )


    if (finalProbPos > finalProbNeg):
        print("\nModel predicts the test query belongs in the POSITVE class")
        writer2.writerow({headList[0]:negTest, headList[1]:str(finalProbNeg), headList[2]:str(finalProbPos), headList[3]:'0'})
    else:
        print("\nModel predicts the test query belongs in the NEGATIVE class")
        writer2.writerow({headList[0]:negTest, headList[1]:str(finalProbNeg), headList[2]:str(finalProbPos), headList[3]:'1'})
        

s_result.close()    
        
#-------------EVALUATION------------------#
print('\nEVALUATION')
        
posReviews = trainPos
negReviews = trainNeg

# put all reviews in one list
theDocs = posReviews + negReviews ;

print('len(posReviews) = ' + str(len(posReviews)))
print('len(negReviews) = ' + str(len(negReviews)))

# create and put all the outcome labels (tags) in one list
labelsPosDocs = [1 for i in posReviews]
labelsNegDocs = [0 for i in negReviews]
labels = labelsPosDocs + labelsNegDocs
#print(labels)


# create the feature vectors using CountVectorizer
cv = CountVectorizer(binary=False,max_df=0.95)
cv.fit_transform(theDocs)
counts = cv.transform(theDocs)   # counts is now a list of feature vectors

x_train, x_test, y_train, y_test  = train_test_split( counts, labels,  test_size=0.4, random_state=1)

print('x_train.shape = ' + str(x_train.shape) )
print('len(y_train) = ' + str(len(y_train)) )
#print('y_train = ' + str(y_train))

print('x_test.shape = ' + str(x_test.shape) )
print('len(y_test) = ' + str(len(y_test)) )

array_x_train = x_train.toarray()


dense_xtrain = x_train.todense()
print("dense_xtrain = ")
print(len(dense_xtrain))


print("\n\n")

print("Training NB Model...")
model_multinomialNB = MultinomialNB()
model_multinomialNB.fit(x_train,y_train)
y_predicted = model_multinomialNB.predict(x_test)


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_predicted))



CM = confusion_matrix(y_test, y_predicted)
TN = CM[0][0]
FN = CM[1][0]
TP = CM[1][1]
FP = CM[0][1]

print("TN = " + str(TN))
print("TP = " + str(TP))
print("FN = " + str(FN))
print("FP = " + str(FP))

print("\naccuracy = ", end = '')
print(accuracy_score(y_test, y_predicted))
print("hand calculated accuracy = " + str( (TN+TP)/(TN+TP+FN+FP)))


print("\nprecision (AKA postive predictive value) = ", end = '' )
print(precision_score(y_test, y_predicted))
# print("hand calculated precision = " + str( TP/(TP+FP)) )
print("hand calculated precision = TP/(TP+FP) = " + str( TP/(TP+FP)) )

print("\nrecall (AKA true postive rate) = " , end = '')
print(recall_score(y_test, y_predicted))
print("hand calculated recall = TP/TP+FN = " + str( TP/(TP+FN)) )

## Now get the equivalent for the negative
# One way you could get it is simple invert the labels and then get accuracy/recall, 
#    but that seems weird to me.  Instead hand calculate:

# print("\nnegative precision (AKA negative predictive value) = ", end = '' )
# print("hand calculated negative precision = TN/(TN+FN) = " + str( TN/(TN+FN)) )

print("\nnegative precision (AKA negative predictive value) = TN/(TN+FN) = " + str( TN/(TN+FN)) )

# print("\nnegative recall (or the true negative rate) = " , end = '')
# print("hand calculated negative recall = TN/(TN+FP) = " + str( TN/(TN+FP)) )

print("\nnegative recall (AKA true negative rate) = TN/(TN+FP) = " + str( TN/(TN+FP)) )

print("\n")