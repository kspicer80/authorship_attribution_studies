import os
import string
from strip_headers_and_footers import strip_headers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt

def bag_of_words(words):
	return dict([(word, True) for word in words])

def load_directory(directory):
    documents, authors, titles = [], [], []
    for filename in os.scandir(directory):
        if not filename.name.endswith('.txt'):
            continue
        author, _ = os.path.splitext(filename.name)

        with open(filename.path, encoding='utf-8') as f:
            contents = f.read()
        contents = strip_headers(contents)
        lemmas = contents.lower().split()
        documents.append(' '.join(lemmas))
        authors.append(author[0])
        title = filename.name.replace('.txt', '').split('_')[1]
        titles.append(f"{title}")

    return documents, authors, titles

training_data_folder = './datasets/'

documents, authors, titles = load_directory(training_data_folder)
X = documents
y = authors

X_train, X_test, y_train, y_test = train_test_split(X, y
                                  ,test_size=0.2, random_state=1234)
bow_transformer=CountVectorizer(analyzer=bag_of_words).fit(X_train)
text_bow_train=bow_transformer.transform(X_train)
text_bow_test=bow_transformer.transform(X_test)

model = MultinomialNB()
model = model.fit(text_bow_train, y_train)
results = model.score(text_bow_train, y_train)
print(results)

validation_results = model.score(text_bow_test, y_test)
print(validation_results)

predictions = model.predict(text_bow_test)
print(classification_report(y_test, predictions))

def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
		print('normalized confusion matrix')
	else:
		print('confusion matrix, without normalization')
	
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", 	color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('true label')
	plt.xlabel('predicted label')
	plt.show()

cm = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cm, classes=['j', 'c'], normalize=False, title='confusion matrix')
print(cm)
