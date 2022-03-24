import numpy as np
import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from strip_headers_and_footers import strip_headers

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

text_zips = list(zip(documents, authors, titles))
df = pd.DataFrame(text_zips, columns=['text', 'author', 'title'])
#print(df.head())

training_data, testing_data = train_test_split(df, random_state=2000)
Y_train = training_data['author'].values
Y_test = testing_data['author'].values

def extract_features(df,field,training_data,testing_data,type="binary"):
    """Extract features using different methods"""
    logging.info("Extracting features and creating vocabulary...")
    if "binary" in type:
        # BINARY FEATURE REPRESENTATION
        cv= CountVectorizer(binary=True, max_df=0.95)
        cv.fit_transform(training_data[field].values)
        train_feature_set=cv.transform(training_data[field].values)
        test_feature_set=cv.transform(testing_data[field].values)
        return train_feature_set,test_feature_set,cv
    elif "counts" in type:
        # COUNT BASED FEATURE REPRESENTATION
        cv= CountVectorizer(binary=False, max_df=0.95)
        cv.fit_transform(training_data[field].values)
        train_feature_set=cv.transform(training_data[field].values)
        test_feature_set=cv.transform(testing_data[field].values)        
        return train_feature_set,test_feature_set,cv
    else:    
        # TF-IDF BASED FEATURE REPRESENTATION
        tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
        tfidf_vectorizer.fit_transform(training_data[field].values)
        train_feature_set=tfidf_vectorizer.transform(training_data[field].values)
        test_feature_set=tfidf_vectorizer.transform(testing_data[field].values)
        return train_feature_set,test_feature_set,tfidf_vectorizer

X_train, X_test, feature_transformer = extract_features(df, 'text', training_data, testing_data, type='binary')

logging.info("Training a Logistic Regression Model ...")
scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear', random_state=0, C=5, penalty='l2', max_iter=1000)
model = scikit_log_reg.fit(X_train, Y_train)

def get_top_k_predictions(model,X_test,k):
    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)
    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:,-k:]
    # GET CATEGORY OF PREDICTIONS
    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
    # REVERSE CATEGORIES - DESCENDING ORDER OF IMPORTANCE
    preds=[ item[::-1] for item in preds]
    return preds

def train_model(df,field="text",feature_rep="binary",top_k=3):
    logging.info("Starting model training...")
    # GET A TRAIN TEST SPLIT (set seed for consistent results)
    training_data, testing_data = train_test_split(df,random_state = 2000,)
    # GET LABELS
    Y_train=training_data['author'].values
    Y_test=testing_data['author'].values
    # GET FEATURES
    X_train,X_test,feature_transformer=extract_features(df,field,training_data,testing_data,type=feature_rep)
    # INIT LOGISTIC REGRESSION CLASSIFIER
    logging.info("Training a Logistic Regression Model...")
    scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
    model=scikit_log_reg.fit(X_train,Y_train)
    # GET TOP K PREDICTIONS
    preds=get_top_k_predictions(model,X_test,top_k)
    # GET PREDICTED VALUES AND GROUND TRUTH INTO A LIST OF LISTS - for ease of evaluation
    eval_items=collect_preds(Y_test,preds)
    # GET EVALUATION NUMBERS ON TEST SET -- HOW DID WE DO?
    logging.info("Starting evaluation...")
    accuracy=compute_accuracy(eval_items)
    mrr_at_k=compute_mrr_at_k(eval_items)
    logging.info("Done training and evaluation.")
    return model,feature_transformer,accuracy,mrr_at_k
    
def compute_mrr_at_k(items:list):
    """Compute the MRR (average RR) at cutoff k"""
    rr_total = 0
    for item in items:   
        rr_at_k = _reciprocal_rank(item[0],item[1])
        rr_total = rr_total + rr_at_k
        mrr = rr_total / 1/float(len(items))
    return mrr

def collect_preds(Y_test,Y_preds):
    """Collect all predictions and ground truth"""
    pred_gold_list=[[[Y_test[idx]],pred] for idx,pred in enumerate(Y_preds)]
    return pred_gold_list
             
def compute_accuracy(eval_items:list):
    correct=0
    total=0
    for item in eval_items:
        true_pred=item[0]
        machine_pred=set(item[1])
        for cat in true_pred:
            if cat in machine_pred:
                correct+=1
                break
    accuracy=correct/float(len(eval_items))
    return accuracy

def _reciprocal_rank(true_labels: list, machine_preds: list):
    """Compute the reciprocal rank at cutoff k"""
    # add index to list only if machine predicted label exists in true labels
    tp_pos_list = [(idx + 1) for idx, r in enumerate(machine_preds) if r in true_labels]
    rr = 0
    if len(tp_pos_list) > 0:
        # for RR we need position of first correct item
        first_pos_list = tp_pos_list[0]
        # rr = 1/rank
        rr = 1 / float(first_pos_list)
    return rr

preds = get_top_k_predictions(model, X_test, 15)
eval_items = collect_preds(Y_test, preds)

logging.info('Starting Evaluation ...')
accuracy = compute_accuracy(eval_items)
mrr_at_k = compute_mrr_at_k(eval_items)

field = 'text'
feature_rep = 'binary'
top_k = 5

model, transformer, accuracy, mrr_at_k = train_model(df, field=field, feature_rep=feature_rep, top_k=top_k)
print("\nAccuracy={0}; MRR={1}".format(accuracy,mrr_at_k))

field = 'text'
feature_rep = 'tfidf'
top_k = 5

model, transformer, accuracy, mrr_at_k = train_model(df, field=field, feature_rep=feature_rep, top_k=top_k)
print("\nAccuracy={0}; MRR={1}".format(accuracy,mrr_at_k))

with open('./testing_data/c_pauls_case.txt', encoding='utf-8') as f:
    test_file = f.read()

test_features = transformer.transform([test_file])
print(get_top_k_predictions(model, test_features, 1))

with open('./testing_data/j_mate_of_the_daylight.txt') as f:
    test_file_1 = f.read()

test_features_1 = transformer.transform([test_file_1])
print(get_top_k_predictions(model, test_features_1, 1))
