# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Willa Cather Investigations

# First things first, we grabbed a copy of Cather's *O Pioneers!* from gutenberg.org [here](https://www.gutenberg.org/files/24/24-h/24-h.htm). We then utilized Jonathan Reeve's nice little ["chapterize"](https://github.com/JonathanReeve/chapterize) library to split the text up into chapters. This CLI tool separates all of the chapters into their own separate .txt files.

# Importing the needed libraries
import json
import glob
import re
import os
import string
import numpy as np
import itertools
from nltk import word_tokenize, sent_tokenize
import codecs
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

# Reeve's chapterize chops off the Gutenberg header and footer material—simple code to do that manually (which is how I did it the first time around) looks something like this:
#
# ``` python
# with open(name) as fn:
#         content = fn.read()
#         start = re.search(r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*|\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", content, re.IGNORECASE).end()
#         stop = re.search(r"End of the Project Gutenberg EBook|End of this Project Gutenberg Ebook", content, re.IGNORECASE).start()
#         final_text = content[start:stop]
# ```
#
#

# Next we want to take all the separated text files and then pair them up with their respective chapter numbers. _O Pioneers!_ is divided into four parts with chapters named by Roman numerals. I keyed them as "1.1," "1.2," etc. all the way to "5.3" to denote, Chapter 3 of Part 5. We'll put these in a list:

chapter_parts = ["1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "2.10", "2.11", "2.12", "3.1", "3.2", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "5.1", "5.2", "5.3"]

file_list = glob.glob(r"G:\My Drive\DHStuff\projects\willa_cather\texts\network_o_pioneers-chapters\*.txt")
file_list = sorted(file_list)

# Just checking to see that all the files are there
file_list

# Now we'll read in each of the files and then zip together the chapter numbers and the text of each chapter. We'll also output it into a .json file that we'll use for the visualizations.

# +
chapters_text = []

for file in file_list:
    with open(file, encoding="utf-8") as f_input:
        text = f_input.read()
        text =  text.replace("\n", " ")
        text = text.replace('"', "'")
        chapters_text.append(text)

pioneers_dict = dict(zip(chapter_parts, chapters_text))

with open('pioneers.json', 'w') as json_file:
    json.dump(pioneers_dict, json_file)
# -

# Since we're interested in producing some network graphs of characters in the novel, I hopped over to Wikipedia to grab a short [character list](https://en.wikipedia.org/wiki/O_Pioneers!#Characters). I was also curious to see if the [spacy](https://spacy.io) NER would pick up any other named entities.

# +
from pprint import pprint
import spacy
nlp = spacy.load('en_core_web_lg')
from collections import Counter

file = open(r'.\texts\o_pioneers.txt', encoding='utf-8').read()
text = nlp(file)
named_entities = [ent for ent in text.ents if ent.label_ == 'PERSON']
pprint(Counter(named_entities).most_common(50))


# -

# It seems like the Wikipedia list was probably good enough. 

# +
# Now we want to write some functions—first one will find "connections" between characters: if two characters' names appear in a chapter, we'll call them "connected":

def connected_characters(text, character_list):
    possible_pairs = list(itertools.combinations(character_list, 2))
    connected = dict.fromkeys(possible_pairs, 0)
    for title, chapter in text['chapters'].items():
        for sent in sent_tokenize(chapter):
            for pair in possible_pairs:
                if pair[0] in sent and pair[1] in sent:
                    connected[pair] += 1
    return connected


# +
# Now we'll read in the .json file and draw a network graph of connected characters:

with codecs.open('pioneers_1.json', 'r', 'utf-8') as data:
    text = json.load(data, strict=False)
    cast = text['cast']

    G = nx.Graph()
    G.name = "The Social Network of O Pioneers!"

    pairs = connected_characters(text, cast)
    for pair, wgt in pairs.items():
        if wgt > 0:
            G.add_edge(pair[0], pair[1], weight=wgt)

    D = nx.ego_graph(G, "Milly")
    edges, weights =  zip(*nx.get_edge_attributes(D, 'weight').items())
    pos = nx.spring_layout(D, k=.5, iterations=40)
    nx.draw(D, pos, node_color='gold', node_size=50, edgelist=edges, width=.5, edge_color='blue', with_labels=True, font_size=12)
    plt.show()

# -

# Another slightly different way to visualize connections between characters:

def character_matrix(text, cast):
    matrix = []
    for first in cast:
        row = []
        for second in cast:
            count = 0
            for title, chapter in text['chapters'].items():
                for sent in sent_tokenize(chapter):
                    if first in sent and second in sent:
                        count += 1
            row.append(count)
        matrix.append(row)
    return matrix


# +
# Plot it:

mtx = character_matrix(text, cast)
plt.figure(figsize=(20,10))
fig, ax = plt.subplots()
fig.suptitle('Character Co-occurence in O Pioneers!', fontsize=12)
fig.subplots_adjust(wspace=.75)
n = len(cast)
x_tick_marks = np.arange(n)
y_tick_marks = np.arange(n)

ax1 = plt.subplot(121)
ax1.set_xticks(x_tick_marks)
ax1.set_yticks(y_tick_marks)
ax1.set_xticklabels(cast, fontsize=8, rotation=90)
ax1.set_yticklabels(cast, fontsize=8)
ax1.xaxis.tick_top()
ax1.set_xlabel('By Frequency')
plt.imshow(mtx,
           norm=matplotlib.colors.LogNorm(),
           interpolation='nearest',
           cmap='RdPu')


alpha_cast = sorted(cast)
alpha_mtx = character_matrix(text, alpha_cast)

ax2 = plt.subplot(122)
ax2.set_xticks(x_tick_marks)
ax2.set_yticks(y_tick_marks)
ax2.set_xticklabels(alpha_cast, fontsize=8, rotation=90)
ax2.set_yticklabels(alpha_cast, fontsize=8)
ax2.xaxis.tick_top()
ax2.set_xlabel("Alphabetically")
plt.imshow(alpha_mtx,
           norm=matplotlib.colors.LogNorm(),
           interpolation='nearest',
           cmap='Blues')
plt.show()
# -

# How about a simple visualization of when in the novel characters show up/appear?

pioneers_words = []
headings = []
chap_lengths = []

# +
for heading, chapter in text['chapters'].items():
    headings.append(heading)
    for sent in sent_tokenize(chapter):
        for word in word_tokenize(sent):
            pioneers_words.append(word)
    chap_lengths.append(len(pioneers_words))

chap_starts = [0] + chap_lengths[:-1]
chap_marks = list(zip(chap_starts, headings))
# -

cast.reverse()
points = []
for y in range(len(cast)):
    for x in range(len(pioneers_words)):
        if len(cast[y].split()) == 1:
            if cast[y] == pioneers_words[x]:
                points.append((x, y))
            else:
                if cast[y] == ' '.join((pioneers_words[x-1], pioneers_words[x])):
                    points.append((x, y))
    if points:
        x, y = list(zip(*points))
    else:
        x = y = ()

# Plot a "dispersion plot" of sorts, but with when characters appear:
fig, ax = plt.subplots(figsize=(15,10))
for chap in chap_marks:
    plt.axvline(x=chap[0], linestyle='-',
        color='gainsboro')
    plt.text(chap[0], -1.55, chap[1], size=7, rotation=90)
plt.plot(x, y, ".", color="mediumorchid", scalex=.1)
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.yticks(list(range(len(cast))), cast, size=8)
plt.ylim(-1, len(cast))
plt.title("Character Mentions in O Pioneers!")
plt.show()

chap[0]

points

# +
import pandas as pd

df = pd.DataFrame.from_records(points, columns=['location_in_text', 'character'])
df.head()
# -

import seaborn as sns
plt.figure(figsize=(15, 10))
plot = sns.stripplot(x='location_in_text', y='character', data=df, palette='Set2', size=10, marker='.', edgecolor='gray', alpha=.50)
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
plt.yticks(list(range(len(cast))), cast, size=8)
plt.ylim(-1, len(cast))
plt.title("Character Mentions in O Pioneers!")

len(cast)

with open("./texts/o_pioneers.txt", encoding="utf-8") as fn:
        content = fn.read()
        start = re.search(r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*|\*\*\* START OF THE PROJECT GUTENBERG EBOOK .* \*\*\*", content, re.IGNORECASE).end()
        stop = re.search(r"End of the Project Gutenberg EBook|End of this Project Gutenberg Ebook", content, re.IGNORECASE).start()
        final_text = content[start:stop]
        final_text = final_text.replace('\n', "")
        final_text = final_text.replace('\n\n', "")

document = nlp(final_text)

words = [token.text for token in document if token.is_stop != True and token.is_punct != True]

word_freq = Counter(words)

common_words = word_freq.most_common(50)
common_words


