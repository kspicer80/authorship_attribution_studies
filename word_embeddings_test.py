import pandas as pd
import spacy
nlp = spacy.load('en_core_web_lg')
from glob import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from numpy import dot
from numpy.linalg import norm
import os
from strip_headers_and_footers import strip_headers
from adjustText import adjust_text
# import word2vec
# import gensim
# from gensim.test.utils import common_texts
# from gensim.models import Word2Vec

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

data_folder = 'cjlw_texts/'

documents, authors, titles = load_directory(data_folder)

cjlwDocs = [nlp(text) for text in documents]

cjlwVecs = [doc.vector for doc in cjlwDocs]

similarities = []
for vec in cjlwDocs:
    this_similarity = [vec.similarity(other) for other in cjlwDocs]
    similarities.append(this_similarity)

#df = pd.DataFrame(similarities, columns = authors, index=authors)
#df[df < 1].idxmax()

pcaOut = PCA(n_components=10).fit_transform(cjlwVecs)
tsneOut = TSNE(n_components=2).fit_transform(pcaOut)

xs, ys = tsneOut[:,0], tsneOut[:,1]
for i in range(len(xs)):
    plt.scatter(xs[i], ys[i])
    plt.annotate((authors[i], titles[i]), (xs[i], ys[i]))
plt.show()
