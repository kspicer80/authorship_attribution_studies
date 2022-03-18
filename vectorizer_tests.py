import os
import re
import numpy as np
import sklearn.feature_extraction.text as text
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import scipy.spatial.distance as scidist
import sklearn.metrics as metrics
from strip_headers_and_footers import strip_headers

class Delta:
    def fit(self, X, y):
        self.train_y = np.array(y)
        self.scaler = preprocessing.StandardScaler(with_mean=False)
        self.train_X = self.scaler.fit_transform(X)
        return self
    
    def predict(self, X, metric='cityblock'):
        X = self.scaler.transform(X)
        dists = scidist.cdist(X, self.train_X, metric=metric)
        return self.train_y[np.argmin(dists, axis=1)]
        
# Loading our data:
def load_directory(directory, max_length):
    documents, authors, titles = [], [], []
    for filename in os.scandir(directory):
        if not filename.name.endswith('.txt'):
            continue
        author, _ = os.path.splitext(filename.name)
        
        with open(filename.path, encoding='utf-8') as f:
            contents = f.read()
        contents = strip_headers(contents)
        lemmas = contents.lower().split()
        start_idx, end_idx, segm_cnt = 0, max_length, 1
        
        while end_idx < len(lemmas):
            documents.append(' '. join(lemmas[start_idx:end_idx]))
            authors.append(author[0])
            title = filename.name.replace('.txt', '').split('_')[1]
            #filename.name.replace('.txt', '').split('_')[0]
            #print(title)
            titles.append(f"{title}-{segm_cnt}")
            
            start_idx += max_length
            end_idx += max_length
            segm_cnt += 1
    
    return documents, authors, titles
    
directory = "./training_data/"

documents, authors, titles = load_directory(directory, 10000)
#print(authors[0])
#print(authors[100])

vectorizer = text.CountVectorizer(max_features=30, token_pattern=r"(?u)\b\w\w+\b")
v_documents = vectorizer.fit_transform(documents).toarray()
#print(v_documents.shape)
#print(vectorizer.get_feature_names()[:10])
v_documents = preprocessing.normalize(v_documents.astype(float), norm='l1')
scaler = preprocessing.StandardScaler()
s_documents = scaler.fit_transform(v_documents)

with open(r'./testing_data/mate_of_the_daylight.txt', 'r', encoding='utf-8') as f:
    single_test_document = f.read()

test_doc = s_documents[0]
distances = [scidist.cityblock(test_doc, train_doc) for train_doc in s_documents[1:]]
#print(authors[np.argmin(distances)+1])

test_size = len(set(authors)) * 2
(train_documents, test_documents, train_authors, test_authors) = model_selection.train_test_split(
    v_documents,
    authors,
    test_size = test_size,
    stratify=authors,
    random_state=1
)

#print(f'N={test_documents.shape[0]} test documents with '
      #f'V{test_documents.shape[1]} features.')

#print(f'N={train_documents.shape[0]} test documents with '
      #f'V{train_documents.shape[1]} features.')
      
scaler = preprocessing.StandardScaler()
scaler.fit(train_documents)
train_documents = scaler.transform(train_documents)
test_documents = scaler.transform(test_documents)

distances = scidist.cdist(test_documents, train_documents, metric='cityblock')

nn_predictions = np.array(train_authors)[np.argmin(distances, axis=1)]
#print(nn_predictions[:3])

delta = Delta()
delta.fit(train_documents, train_authors)
preds = delta.predict(test_documents)

for true, pred in zip(test_authors, preds):
    _connector = 'WHEREAS' if true != pred else 'and'
    #print(f'Observed author is {true} {_connector} {pred} was predicted.') 
    
accuracy = metrics.accuracy_score(preds, test_authors)
#print(f'\nAccuracy of predictions: {accuracy:.1f}')

with open(r'C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\testing_data\mate_of_the_daylight.txt') as f:
    test_doc = f.read()

#lemmas = test_doc.lower().split()
#print(len(lemmas))

v_test_doc = vectorizer.transform([test_doc]).toarray()
v_test_doc = preprocessing.normalize(v_test_doc.astype(float), norm='l1')
print(delta.predict(v_test_doc)[0])

with open(r"C:\Users\KSpicer\Documents\GitHub\cather_jewett_comparisons\testing_data\archbishop.txt", encoding='utf-8') as f:
    test_doc_1 = f.read()

#lemmas_two = test_doc_1.lower().split()
#print(len(lemmas_two))

v_test_doc_1 = vectorizer.transform([test_doc_1]).toarray()
v_test_doc_1 = preprocessing.normalize(v_test_doc_1.astype(float), norm='l1')
print(delta.predict(v_test_doc_1)[0])

pred_different_metric = delta.predict(v_test_doc_1, metric='cosine')
print(pred_different_metric)
