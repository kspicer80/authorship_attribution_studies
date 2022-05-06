import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load('en_core_web_lg')
from strip_headers_and_footers import strip_headers

pauls_case_file = r'cather_jewett\testing_data\c_pauls_case.txt'
mate_file = r'cather_jewett\testing_data\j_mate_of_the_daylight.txt'
#
cather_mean_sentence_lengths = []
jewett_mean_sentence_lengths = []

with open(pauls_case_file, encoding='utf-8') as f:
    cather_data = f.read()
    contents = strip_headers(cather_data)
    cather = nlp(contents)
    print(len(cather))
    all_sents = cather.sents
    sentLengths = [len(sent) for sent in all_sents]
    average_length = np.mean(sentLengths)
    cather_mean_sentence_lengths.append(average_length)

with open(mate_file, encoding='utf-8') as f:
    jewett_data = f.read()
    contents = strip_headers(jewett_data)
    jewett = nlp(contents)
    print(len(jewett))
    all_sents = jewett.sents
    sentLengths = [len(sent) for sent in all_sents]
    average_length = np.mean(sentLengths)
    jewett_mean_sentence_lengths.append(average_length)
    
print(cather_mean_sentence_lengths, jewett_mean_sentence_lengths)
