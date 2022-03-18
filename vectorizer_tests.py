import os
import re
import sklearn.feature_extraction.text as text
import sklearn.preprocessing as preprocessing
from strip_headers_and_footers import strip_headers

# Loading our data:
def load_directory(directory, max_length):
    documents, authors, titles = [], [], []
    for filename in os.scandir(directory):
        if not filename.name.endswith('.txt'):
            continue
        author, _ = os.path.splitext(filename.name)
        
        with open(filename.path) as f:
            contents = f.read()
        lemmas = contents.lower().split()
        start_idx, end_idx, segm_cnt = 0, max_length, 1
        
        while end_idx < len(lemmas):
            documents.append(' '. join(lemmas[start_idx:end_idx]))
            authors.append(author[0])
            title = filename.name.replace('.txt', '').split('_')[1]
            titles.append(f"{title}-{segm_cnt}")
            
            start_idx += max_length
            end_idx += max_length
            segm_cnt += 1
    
    return documents, authors, titles
    
directory = "./training_data/"

for root, subdirectories, files in os.walk(directory):
    for subdirectory in subdirectories:
        for filename in os.scandir(subdirectory):
            if not filename.name.endswith('.txt'):
                continue
            author = subdirectory
    
    for subdirectory in subdirectories:
        print(os.path.join(root, subdirectory))
    for file in files:
        print(os.path.join(root, file))

#for filename in os.scandir(directory):
    #print(filename)



documents, authors, titles = load_directory(directory, 10000)




#cather_file_list = glob.glob(r'.\training_data\cather\*.txt')
#print(cather_file_list)



def my_load_directory(author_name, file_list, max_length):
    documents, author, titles = [], [], []    
    for filename in os.scandir(directory):
        if not filename.name.endswith('.txt'):
            continue
        author, _ = os.path.splitext(filename.name)
    for file in file_list:
        with open(file, encoding='utf-8') as f:
            contents = f.read()
            contents = strip_headers(contents)
            lemmas = contents.lower().split()
            start_idx, end_idx, segm_cnt = 0, max_length, 1        
        while end_idx < len(lemmas):
            documents.append(' '. join(lemmas[start_idx:end_idx]))
            author.append(author_name)
            title = file
            titles.append(f"{title}-{segm_cnt}")            
            start_idx += max_length
            end_idx += max_length
            segm_cnt += 1
    
    return documents, author, titles

#documents, author, titles = load_directory('cather', cather_file_list, 10000)
#print(len(documents))
#print(author)
#print(titles)

#vectorizer = text.CountVectorizer(max_features=30, token_pattern=r"(?u)\b\w\w+\b")
#v_documents = vectorizer.fit_transform(documents).toarray()
##print(v_documents.shape)
##print(vectorizer.get_feature_names()[:10])
#v_documents = preprocessing.normalize(v_documents.astype(float), norm='l1')
