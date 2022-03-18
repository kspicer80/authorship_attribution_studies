import os
import re
import matplotlib.pyplot as plt
import numpy as np
import sklearn.feature_extraction.text as text
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection
import scipy.spatial.distance as scidist
import scipy.cluster.hierarchy as hierarchy
import sklearn.metrics as metrics
from strip_headers_and_footers import strip_headers

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
            title = filename.name.replace('.txt', '').split('_')[1:]
            titles.append(f"{title}-{segm_cnt}")
            
            start_idx += max_length
            end_idx += max_length
            segm_cnt += 1
    
    return documents, authors, titles
    
directory = "./training_data/"

documents, authors, titles = load_directory(directory, 20000)

vectorizer = text.CountVectorizer(max_features=30, token_pattern=r"(?u)\b\w\w+\b")
v_documents = vectorizer.fit_transform(documents).toarray()
v_documents = preprocessing.normalize(v_documents.astype(float), norm='l1')
scaler = preprocessing.StandardScaler()
s_documents = scaler.fit_transform(v_documents)

print(f'N={v_documents.shape[0]} documents with '
      f'V={v_documents.shape[1]} features.')
      
dm = scidist.pdist(v_documents, 'cityblock')
linkage_object = hierarchy.linkage(dm, method='complete')

def plot_tree(linkage_object, labels, figsize=(15, 15), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    with plt.rc_context({'lines.linewidth': 1.0}):
        hierarchy.dendrogram(
            linkage_object,
            labels=labels,
            ax=ax,
            link_color_func=lambda c: 'black',
            leaf_font_size=10,
            leaf_rotation=90)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    for s in ax.spines.values():
        s.set_visible(False)
    plt.show()

plot_tree(linkage_object, titles)
