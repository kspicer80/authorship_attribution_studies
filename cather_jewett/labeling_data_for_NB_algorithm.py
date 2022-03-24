import glob
import pandas as pd
import collections
import os
import re
import math
import string

class cather_or_jewett_detector(object):
	def clean(self, s):
		translator = str.maketrans("", "", string.punctuation)
		return s.translate(translator)
		
	def tokenize(self, text):
		text = self.clean(text).lower()
		return re.split("\W+", text)
	
	def get_word_counts(self, words):
		word_counts = {}
		for word in words:
			word_counts[word] = word_counts.get(word, 0.0) + 1.0
		return word_counts
		
	def fit(self, X, Y):
		self.num_texts = {}
		self.log_class_priors = {}
		self.word_counts = {}
		self.vocab = set()
		
		n = len(X)
		self.num_texts['cather'] = sum(1 for label in Y if label == 0)
		self.num_texts['jewett'] = sum(1 for label in Y if label == 1)
		self.log_class_priors['cather'] = math.log(self.num_texts['cather']/n)
		self.log_class_priors['jewett'] = math.log(self.num_texts['jewett']/n)
		self.word_counts['cather'] = {}
		self.word_counts['jewett'] = {}
		
		for x, y in zip(X, Y):
			c = 'cather' if y == 0 else 'jewett'
			counts = self.get_word_counts(self.tokenize(x))
			for word, count in counts.items():
				if word not in self.vocab:
					self.vocab.add(word)
				if word not in self.word_counts[c]:
					self.word_counts[c][word] = 0.0
				
				self.word_counts[c][word] += count
	
	def predict(self, X):
		result = []
		for x in X:
			counts = self.get_word_counts(self.tokenize(x))
			cather_score = 0
			jewett_score = 0
			for word, _ in counts.items():
				if word not in self.vocab: continue
				
				log_w_given_cather = math.log((self.word_counts['cather'].get(word, 0.0) + 1)/(self.num_texts['cather'] + len(self.vocab)))
				log_w_given_jewett = math.log((self.word_counts['jewett'].get(word, 0.0) + 1)/(self.num_texts['jewett'] + len(self.vocab))) 
				
				cather_score += log_w_given_cather
				jewett_score += log_w_given_jewett
				
			cather_score += self.log_class_priors['cather']
			jewett_score += self.log_class_priors['jewett']
			
			if cather_score > jewett_score:
				result.append(0)
			else:
				result.append(1)
		return result

#c_file_list = glob.glob('/Volumes/GoogleDrive-113389011671541578812/My Drive/DHStuff/projects/willa_cather/data_folder/cather/*.txt')
#c_file_list = sorted(c_file_list)

#j_file_list = glob.glob('/Volumes/GoogleDrive-113389011671541578812/My Drive/DHStuff/projects/willa_cather/data_folder/jewett/*.txt')
#j_file_list = sorted(j_file_list)

data_directory = r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\training_data'
target_names = ['cather', 'jewett']

cather_files = glob.glob(r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\training_data\train0\*.txt')
jewett_files = glob.glob(r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\training_data\train1\*.txt')
# 0 for Cather's texts; 1 for Jewett's texts

def get_data(data_directory):
	data = []
	target = []
	for file in cather_files:
		with open(file, encoding='utf-8') as f:
			data.append(f.read())
			target.append(0)
	for file in jewett_files:
		with open(file, encoding='utf-8') as f:
			data.append(f.read())
			target.append(1)
	return data, target

X, y = get_data(data_directory)

testing_data_file = r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\testing_data\jewett\mate_of_the_daylight.txt'
print(len(X[4]))
print(type(X))
print((y[4]))
print(y)
print(len(X)*.8)
cj_classifier = cather_or_jewett_detector()
cj_classifier.fit(X[5:], y[:])
pred = cj_classifier.predict(testing_data_file)
true = y
#
#
accuracy = sum(1 for i in range(len(pred)) if pred[i] == true[i]) /float(len(pred))
print("{0:.4f}".format(accuracy))

#test = get_data(data_directory)

#c_file_list = glob.glob(r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\training_data\cather\*.txt')
#j_file_list = glob.glob(r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\training#_data\jewett\*.txt')
#
#cather_data = {}
#
#for file in c_file_list:
    #with open(file, 'r', encoding='utf-8') as f:
        #data = f.read()
    #cather_data[data] = 0
#
#jewett_data = {}
#
#for file in j_file_list:
    #with open(file, 'r', encoding='utf-8') as f:
        #data = f.read()
    #jewett_data[data] = 1
#print(len(jewett_data))






#train = {**cather_data, **jewett_data}
#reversed = {value: key for key, value in train.items()}
#print(len(reversed))
#
#
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.pipeline import make_pipeline
#
#model = make_pipeline(TfidfVectorizer(), MultinomialNB())
