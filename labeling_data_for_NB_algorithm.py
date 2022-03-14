import glob
import pandas as pd
import collections

########################################Functions We Need:#############################
def bag_of_words(words):
	return dict([(word, True) for word in words])

def train_test_feats(label, instances, featx=bag_of_words, fraction=0.75):
	labeled_instances = [(featx(i), label) for i in instances]

	if fraction != 1.0:
		l = len(instances)
		cutoff = int(math.ceil(l * fraction))
		return labeled_instances[:cutoff], labeled_instances[cutoff:]
	else:
		return labeled_instances, labeled_instances

def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats

################################################################################
c_file_list = glob.glob('/Volumes/GoogleDrive-113389011671541578812/My Drive/DHStuff/projects/willa_cather/data_folder/cather/*.txt')
c_file_list = sorted(c_file_list)

j_file_list = glob.glob('/Volumes/GoogleDrive-113389011671541578812/My Drive/DHStuff/projects/willa_cather/data_folder/jewett/*.txt')
j_file_list = sorted(j_file_list)

cather_data = {}

for file in c_file_list:
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()
    cather_data[data] = 0

jewett_data = {}

for file in j_file_list:
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()
    jewett_data[data] = 1
#print(len(jewett_data))

train = {**cather_data, **jewett_data}
print(train[1])

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
