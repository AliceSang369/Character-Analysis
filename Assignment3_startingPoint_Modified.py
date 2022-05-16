import nltk
from urllib import request
from nltk.tokenize import RegexpTokenizer
from nltk.util import bigrams
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize


# NOTE:  this file generates two files:  "tempOut-charName" and "tempAnnotated-charName" for 
# each charname you use in the list "characters" below as well as the file "tempOut-FullText"

# tokenizer = RegexpTokenizer(r'\w+')   # use NOT alphanumeric as token separator

# first create the in-memory postive/negative lexicons from files
wordsPositive = set()
fpp = open("positiveWords.txt",mode='r',encoding="ISO-8859-1")
for line in fpp:
	aStr = line.replace('\n','')
	wordsPositive.add(aStr)
# print("len(wordsPositive) = " + str(len(wordsPositive)))

wordsNegative = set()
fpn = open("negativeWords.txt",mode='r',encoding="ISO-8859-1")
for line in fpn:
	aStr = line.replace('\n','')
	line.replace('\n','')
	wordsNegative.add(aStr)
# print("len(wordsNegative) = " + str(len(wordsNegative)))



# Open the book and read in raw removing new-line and carriage return chars
# Great Expectations, by Charles Dickens, 
# =============================================================================
# url = "https://www.gutenberg.org/files/1400/1400-0.txt"
# response = request.urlopen(url)   # open the web page and get the text
# raw = response.read().decode('utf8')   # this is the raw text
# raw = str(raw)
# =============================================================================

# Note - if you want to converts all the different types of unicode quote types (left/right single/double etc) 
#   into ascii quotes, you can use unidecode.  You likely will need to "pip install unidecode'
# https://pypi.org/project/Unidecode/
with open ('Harry_Potter_1.txt', 'r') as rf:
    raw = rf.read()
# =============================================================================
# from unidecode import unidecode
# raw = unidecode(raw)
# =============================================================================

raw = raw.replace('\n','')  # remove all the newline characters
raw = raw.replace('\r',' ')  # replace carriage return characters with a space
fp = open("tempOut-FullText", "w")
fp.write(raw)
fp.close()


# raw is now a giant ascii string holding the entire book
# Time to break it into sentences

tokens = word_tokenize(str(raw))  # tokenize into a list of sentences
sentences = sent_tokenize(str(raw))  # tokenize into a list of sentences
print("\n\nGreat Expectations: len(raw), len(tokens), len(sentences) = " + str(len(raw)) + "," + str(len(tokens)) + "," + str(len(sentences)))


# specify your list of character names you want to analyze
characters = ['Dumbledore', 'Snape'] #'Quirrell'

# creates a dictionay of lists based on using character name as key
sentsByChars = {}
for character in characters:
	sentsByChars[character] = []

# note, the above three lines are equivalent to the following statement (assuming the 
#    same character names) but is less error prone as you don't have to type in the names twice
# sentsByChars = {'Pip':[], 'Miss Havisham':[], 'Estella':[], 'Jaggers':[], 'Gargery':[], 'Drummle':[]}

# Go through all the sentences, and if a sentence contains a character name add it to that dictionary list
# for that character.  Note - a sentence can be placed in multiple lists if multiple charcter names are in the sentence
for s in sentences:
	for name in characters:
		index = s.find(name)
		if index > 0:  # then found
			# print(name)
			sentsByChars[name].append(s)

# Now have the sentences for each character of interest in the dictionary data structure

# For each of the novel characters, count how many correspnding sentences are postive, negative, or neutral
# where "positive" means a sentence contains more words in the postive lexicon than words in the negative lexicon, etc.
# Also, while iterating, lets alls write out a file of counts with the sentence for each novel character
for aKey in sentsByChars.keys():
	# print(aKey + ":" + 'len[aKey] = ' + str( len(sentsByChars[aKey])) )
	fp = open('tempAnnotated-' + aKey, 'w')
	posSentences = 0
	negSentences = 0
	neutralSentences = 0
	for s in sentsByChars[aKey]:
		if (len(s) < 2):
			print("len(s) < 2")
		posCount = 0
		negCount = 0
		tokens = word_tokenize(s)  # word tokenize the sentence
		for t in tokens:   # count how many words are positive/negative
			if t in wordsPositive:
				posCount += 1
			if t in wordsNegative:
				negCount += 1
		if (posCount > negCount):   # update the running total of postive/negative/neutral sentences
			posSentences += 1
			fp.write("P ")   # Positive
		elif (posCount < negCount):
			negSentences += 1
			fp.write("N ")    # Negative
		else:
			neutralSentences += 1
			fp.write("I ")    # Impartial (i.e. neither more postive nor more negative)
		fp.write(str(posCount) + " " + str(negCount) + " " + s + "\n")  # write out the pos/neg counts and the sentence
	# print out the results
	print("\n" + aKey + ": posSentences,negSentences,neutralSentences = " + str(posSentences) + ", " + str(negSentences) + ", " + str(neutralSentences))
	fp.close()

	print("Sentence percentages (pos,neg,neutral) = " +
	"{:.2f}".format(100.0*posSentences/len(sentsByChars[aKey])) + ", " + 
	"{:.2f}".format(100.0*negSentences/len(sentsByChars[aKey])) + ", " + 
	"{:.2f}".format(100.0*neutralSentences/len(sentsByChars[aKey])) )



# If desired, write out to a file the sentences for each character of interest
# Unlike the tempAnnotated-ZZZ files, thes do not have the pos/neg word counts every sentence
#remove all character names.
import re
for aKey in sentsByChars.keys():
	fp = open("noName-" + aKey, "w")
	for s in sentsByChars[aKey]:
		print(s)
		no_name = str(s).replace(str(aKey), '')
		print(no_name)
		print('\n')
		fp.write(no_name)
		fp.write("\n")
	fp.close()






