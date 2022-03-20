import os
import string
from strip_headers_and_footers import strip_headers
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

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
        documents.append(lemmas) 
        #documents.append(' '.join(lemmas))
        authors.append(author[0])
        title = filename.name.replace('.txt', '').split('_')[1]
        titles.append(f"{title}")

    return documents, authors, titles

training_data_folder = './training_data/'
testing_data_folder = './testing_data/'

documents, authors, titles = load_directory(training_data_folder)
X = documents
y = authors

X_train, X_test, y_train, y_test = train_test_split(X, y
                                  ,test_size=0.2, random_state=1234)

bow_transformer=CountVectorizer(analyzer=bag_of_words).fit(X_train)
text_bow_train=bow_transformer.transform(X_train)
text_bow_test=bow_transformer.transform(X_test)
