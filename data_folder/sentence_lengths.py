import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load('en_core_web_lg')

#with open('sun_also_rises.txt') as f:
    #text = f.read()
#
#nlp_text = nlp(text)
#all_sents = list(nlp_text.sents)
##print(len(all_sents))
#
##averages = []
##for lengths in sentLengths:
    ##total = sum(lengths)
    ##num = len(lengths)
    ##average = total/num
    ##averages.append(averages)
##print(averages)
#
#print(np.mean(sentLengths))

#cather_list = glob.glob('/Volumes/GoogleDrive-113389011671541578812/My Drive/DHStuff/projects/cather_jewett_comparisons/data_folder/cather/*.txt')
#jewett_list = glob.glob('/Volumes/GoogleDrive-113389011671541578812/My Drive/DHStuff/projects/cather_jewett_comparisons/data_folder/jewett/*.txt')

cather_file_names = glob.glob('cather/*.txt')
cather_file_names = sorted(cather_file_names)
print(cather_file_names)
cather_labels = [filename.split('/')[1] for filename in cather_file_names]
cather_labels = [filename.split('.')[0] for filename in cather_labels]
print(cather_labels)

jewett_file_names = glob.glob('jewett/*.txt')
jewett_file_names = sorted(jewett_file_names)
print(jewett_file_names)
jewett_labels = [filename.split('/')[1] for filename in jewett_file_names]
jewett_labels = [filename.split('.')[0] for filename in jewett_labels]

#for filename in jewett_file_names:
    #with open(filename, encoding='utf-8') as f:
        #file = f.read()
        #cather = nlp(file)
        #all_sents = cather.sents
        #sentLengths = [len(sent) for sent in all_sents]
        #average_length = np.mean(sentLengths)
        #lengths.append(average_length)

lengths_of_cather_sentences = [16.534602076124568, 16.19121140142518, 14.928874734607218, 19.26351931330472, 18.182510148849797, 16.172173522139808, 16.9465306122449, 25.85636856368564, 16.44207920792079, 17.65152140988572, 17.69668759471747, 19.880056777856637]
lengths_of_jewett_sentences = [24.871145731905226, 30.50354259350799, 27.356521739130436, 27.604920405209842, 31.91920463126101, 25.74780553679946, 31.169653179190753, 24.75655629139073, 25.91968400263331, 26.365143824027072]

cdict = dict(zip(cather_labels, lengths_of_cather_sentences))
jdict = dict(zip(jewett_labels, lengths_of_jewett_sentences))

plt.bar(range(len(cather_labels)), lengths_of_cather_sentences)
plt.xticks(range(len(cather_labels)), cather_labels, rotation=90)
plt.show()

plt.bar(range(len(jdict)), list(jdict.values()))
plt.xticks(range(len(jdict)), list(jdict.keys()), rotation=90)
plt.show()

#averageLength = [sent/len(sentLengths) for sent in sentLengths]
#print(averageLength)

#[nlp(open(doc, errors='ignore').read()) for doc in cather_file_names]
#sentence_lengths = [np.mean([len(sent) for sent in doc.sents]) for doc in cather]
#pd.Series(sentence_lengths, index=cather_labels).plot(kind='bar')
#plt.show()
#string = r'cather\\a_lost_lady.txt'
#split_string = string.split(r'\\')
#print(split_string[1])


#def read_and_convert(texts):
#for file in cather_list:
    #with open(file, encoding='utf-8') as f:
        #data = f.read()
#
#inaugural = [nlp(open(doc, errors='ignore').read()) for doc in #inauguralFilenames]
#sentLengths = [ np.mean([len(sent) for sent in doc.sents]) for doc in inaugural #]
