import math
from nltk import probability
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.metrics.scores import *
import collections

def bag_of_words(words):
	return dict([(word, True) for word in words])

def bag_of_words_in_set(words, wordset):
	return bag_of_words(set(words) and wordset)

def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))

def word_counts(words):
	return dict(probability.FreqDist((w for w in words)))

def word_counts_in_set(words, wordset):
	return word_counts((w for w in words if w in wordset))

def train_test_feats(label, instances, featx=bag_of_words, fraction=0.75):
	labeled_instances = [(featx(i), label) for i in instances]

	if fraction != 1.0:
		l = len(instances)
		cutoff = int(math.ceil(l * fraction))
		return labeled_instances[:cutoff], labeled_instances[cutoff:]
	else:
		return labeled_instances, labeled_instances

def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq,
   n=200):
     bigram_finder = BigramCollocationFinder.from_words(words)
     bigrams = bigram_finder.nbest(score_fn, n)
     return bag_of_words(words + bigrams)

#print(bag_of_bigrams_words(['the', 'quick', 'brown', 'fox']))
#print(bag_of_words(['the', 'big', 'stupid', 'bear']))
#test = bag_of_words_not_in_set(['the', 'big', 'stupid', 'bear'], ['the'])
#print(test)
#test_1 = bag_of_words_in_set(['the', 'big', 'stupid', 'bear'], ['bear', 'big'])
#print(test_1)

def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    label_feats = collections.defaultdict(list)
    for label in corp.categories():
        for fileid in corp.fileids(categories=[label]):
            feats = feature_detector(corp.words(fileids=[fileid]))
            label_feats[label].append(feats)
    return label_feats

def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats

def precision_recall(classifier, testfeats):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    precisions = {}
    recalls = {}
    for label in classifier.labels():
        precisions[label] = precision(refsets[label], testsets[label])
        recalls[label] = recall(refsets[label], testsets[label])
    return precisions, recalls

def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq, min_score=5):
    word_fd = FreqDist()
    label_word_fd = ConditionalFreqDist()

    for label, words in labelled_words:
        for word in words:
            word_fd[word] += 1
            label_word_fd[label][word] += 1

    n_xx = label_word_fd.N()
    high_info_words = set()

    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        word_scores = collections.defaultdict(int)
        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score
        bestwords = [word for word, score in word_scores.items() if score >= min_score]
        high_info_words |= set(bestwords)
    return high_info_words

#print(movie_reviews.categories())
lfeats = label_feats_from_corpus(movie_reviews)
#print(lfeats.keys())
train_feats, test_feats = split_label_feats(lfeats)
#print(len(train_feats))
#print(len(test_feats))

nb_classifier = NaiveBayesClassifier.train(train_feats)
#print(nb_classifier.labels())

negfeattest = bag_of_words(['this', 'plot', 'was', 'so', 'silly', 'and', 'stupid'])
#print(nb_classifier.classify(negfeattest))

posfeattest = bag_of_words(['this', 'book', 'is', 'so', 'good', 'I', "can't", 'stop', 'reading', 'it'])
#print(nb_classifier.classify(posfeattest))

from nltk.classify.util import accuracy
#print(accuracy(nb_classifier, test_feats))

probs = nb_classifier.prob_classify(test_feats[0][0])
#print(probs.prob('pos'))
#print(test_feats[0][0])

#print(nb_classifier.most_informative_features(n=5))
#print(nb_classifier.show_most_informative_features(n=5))

#dt_classifier = DecisionTreeClassifier.train(train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)
#print(accuracy(dt_classifier, test_feats))

nb_precisions, nb_recalls = precision_recall(nb_classifier, test_feats)
#print(nb_precisions, nb_recalls)

labels = movie_reviews.categories()
labeled_words = [(l, movie_reviews.words(categories=[l])) for l in labels]
high_info_words = set(high_information_words(labeled_words))
feat_det = lambda words: bag_of_words_in_set(words, high_info_words)
lfeats = label_feats_from_corpus(movie_reviews, feature_detector=feat_det)
train_feats, test_feats = split_label_feats(lfeats)

nb_classifier = NaiveBayesClassifier.train(train_feats)
print(accuracy(nb_classifier, test_feats))
nb_precisions, nb_recalls = precision_recall(nb_classifier, test_feats)
print(nb_precisions['pos'], nb_recalls['pos'])
