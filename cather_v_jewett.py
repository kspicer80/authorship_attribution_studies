from keras.preprocessing.text import Tokenizer
from keras import preprocessing
import json
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
import random
from tensorflow import keras
import glob

def get_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    data = data.lower()
    return(data)

def get_index(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return(data)

def create_sents(text):
    sentences = text.split(".")
    return(sentences)

def padding_data(sentences, index, maxlen=25):
    new_sentences = []
    for sentence in sentences:
        sentence = text_to_word_sequence(sentence)
        new_sentence = []
        words = []
        for word in sentence:
            try:
                word = index[word]
            except:
                KeyError
                word = 0
            words.append(word)
        new_sentence.append(words)
        new_sentence = preprocessing.sequence.pad_sequences(new_sentence, maxlen=maxlen, padding='post')
        new_sentences.append(new_sentence[0])
    return(new_sentences)

def create_index(texts, filename):
    words = texts.split()
    tokenizer = Tokenizer(num_words = 100000)
    tokenizer.fit_on_texts(words)
    sequences = tokenizer.texts_to_sequences(words)
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique words.")
    with open(filename, "w") as f:
        json.dump(word_index, f, indent=4)

def reverse_index(word_index):
    reverse_word_index = {value: key for (key, value) in word_index.items()}
    return(reverse_word_index)

def reconst_text(text, reverse_word_index):
    return(" ".join([reverse_word_index.get(i, "?") for i in text]))

def label_data(sentences, label):
    total_chunks = []
    for sentence in sentences:
        total_chunks.append((sentence, label))
    return(total_chunks)

def create_training(total_chunks, cutoff):
    random.shuffle(total_chunks)
    training_data = []
    training_labels = []
    testing_data = []
    testing_labels = []
    test_num = len(total_chunks)*cutoff
    x = 0
    for entry in total_chunks:
        if x > test_num:
            testing_data.append(entry[0])
            testing_labels.append(entry[1])
        else:
            training_data.append(entry[0])
            training_labels.append(entry[1])
        x = x+1
    training_data = np.array(training_data)
    training_labels = np.array(training_labels)
    testing_data = np.array(testing_data)
    testing_labels = np.array(testing_labels)
    return(training_data, training_labels, testing_data, testing_labels)

def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Embedding(15000, 25))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(16, activation='tanh'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return(model)

def train_model(model, tt_data, val_size=.3, epochs=1, batch_size=16):
    vals = int(len(tt_data[0])*val_size)
    training_data = tt_data[0]
    training_labels = tt_data[1]
    testing_data = tt_data[2]
    testing_labels = tt_data[3]

    x_val = training_data[:vals]
    x_train = training_data[vals:]

    y_val = training_labels[:vals]
    y_train = training_labels[vals:]

    fitModel = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=2, shuffle=True)
    print(fitModel.history.keys())
    import matplotlib.pyplot as plt
    plt.plot(fitModel.history['loss'])
    plt.plot(fitModel.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.clf()
    plt.plot(fitModel.history['accuracy'])
    plt.plot(fitModel.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()
    model_results = model.evaluate(testing_data, testing_labels)
    return(model)

def plot_model_loss(model_name, string_1='loss', string_2='val_loss'):
    plt.plot(model_name.history[string_1])
    plt.plot(model_name.history[string_2])
    plt.title('model loss')
    plt.ylabel(string_1)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_model_accuracy(model_name, string_1='accuracy', string_2='val_accuracy'):
    plt.plot(model_name.history[string_1])
    plt.plot(model_name.history[string_2])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()

c_file_list = glob.glob('/Users/spicy.kev/Desktop/cather_jewett_comparisons/data_folder/cather/*.txt')
c_file_list = sorted(c_file_list)

j_file_list = glob.glob('/Users/spicy.kev/Desktop/cather_jewett_comparisons/data_folder/jewett/*.txt')
j_file_list = sorted(j_file_list)

for file in c_file_list:
    c_data = get_data(file)

for file in j_file_list:
    j_data = get_data(file)

all_texts = c_data + j_data

#create_index(all_texts, 'word_index_test.json')

word_index = get_index('/Users/spicy.kev/Desktop/cather_jewett_comparisons/word_index_test.json')
reverse_word_index = reverse_index(word_index)

j_sents = create_sents(j_data)
c_sents = create_sents(c_data)
#print(j_sents[0])
#print(c_sents[0])

j_padded = padding_data(j_sents, word_index, maxlen=25)
c_padded = padding_data(c_sents, word_index, maxlen=25)
#print(j_padded[0])

#print(j_sents[0])
#print(j_padded[0])
#print(reconst_text(j_padded[0], reverse_word_index))

c_labeled = label_data(c_padded, 0)
j_labeled = label_data(j_padded, 1)

all_data = c_labeled + j_labeled
tt_data = create_training(all_data, cutoff=.95)
print(tt_data[1])

model = create_model()
model = train_model(model, tt_data=tt_data, epochs=10, batch_size=16)
#model.save('cj_model')

#model = keras.models.load_model('cj_model')

def test_model(text_chunks, reverse_word_index, model, cutoff=0):
    results = []
    print("Analysis: ")
    for test in text_chunks:
        if len(test) > 2:
            print(test)
            predict = model.predict([test])
            if predict[0] > cutoff:
                print("Prediction: " +str(predict[0]))
                results.append((str(predict[0]), reconst_text(test, reverse_word_index)))
    return(results, cutoff)

def write_test(results, filename, name):
    with open(filename + '.txt', 'w', encoding='utf-8') as f:
        f.write("**********TEST ON**************")
        f.write(f"**********{name}**************")
        for result in results:
            f.write(str(result)+'\n')

#t_file = '/Users/spicy.kev/Desktop/cather_jewett_comparisons/testing_data/jewett/mate_of_the_daylight.txt'
t_file = '/Users/spicy.kev/Desktop/cather_jewett_comparisons/testing_data/hemingway/sun_also_rises.txt'
t_text = get_data(t_file)
t_sents = create_sents(t_text)
t_padded = padding_data(t_sents, word_index, maxlen=25)

test_results = test_model(t_padded, reverse_word_index=reverse_word_index, model=model)
write_test(test_results[0], filename='heminway_results', name='Sun Also Rises')
