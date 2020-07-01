import re

import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model

from attention import AttentionLayer # (https://github.com/thushv89/attention_keras/blob/master/layers/attention.py)

K.clear_session()
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                  device_count={'GPU': 0, 'CPU': 10})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

wikihowAll = pd.read_csv("wikihowAll.csv")

wikihowAll.drop_duplicates(subset=['text'], inplace=True)
wikihowAll.dropna(axis=0, inplace=True)

contractionMapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                      "could've": "could have", "couldn't": "could not",
                      "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                      "hasn't": "has not", "haven't": "have not",
                      "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                      "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                      "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                      "I'm": "I am", "I've": "I have", "i'd": "i would",
                      "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                      "i've": "i have", "isn't": "is not", "it'd": "it would",
                      "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                      "let's": "let us", "ma'am": "madam",
                      "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                      "mightn't've": "might not have", "must've": "must have",
                      "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                      "needn't've": "need not have", "o'clock": "of the clock",
                      "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                      "sha'n't": "shall not", "shan't've": "shall not have",
                      "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                      "she'll've": "she will have", "she's": "she is",
                      "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                      "so've": "so have", "so's": "so as",
                      "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                      "there'd": "there would",
                      "there'd've": "there would have", "there's": "there is", "here's": "here is",
                      "they'd": "they would", "they'd've": "they would have",
                      "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                      "they've": "they have", "to've": "to have",
                      "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                      "we'll've": "we will have", "we're": "we are",
                      "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                      "what're": "what are",
                      "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                      "where'd": "where did", "where's": "where is",
                      "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                      "who've": "who have",
                      "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                      "won't've": "will not have",
                      "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                      "y'all": "you all",
                      "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                      "y'all've": "you all have",
                      "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                      "you'll've": "you will have",
                      "you're": "you are", "you've": "you have"}

stopWords = set(stopwords.words('english'))


def preprocess(text, num):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"', '', newString)
    newString = ' '.join([contractionMapping[i] if i in contractionMapping else i for i in newString.split(" ")])
    newString = re.sub(r"'s\b", "", newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = re.sub('[m]{2,}', 'mm', newString)
    if (num == 0):
        tokens = [i for i in newString.split() if not i in stopWords]
    else:
        tokens = newString.split()
    longWords = []
    for i in tokens:
        if len(i) > 1:
            longWords.append(i)
    return (" ".join(longWords)).strip()


cleanedText = []
for i in wikihowAll['text']:
    cleanedText.append(preprocess(i, 0))

cleanedSummary = []
for i in wikihowAll['title']:
    cleanedSummary.append(preprocess(i, 1))

wikihowAll['cleanedText'] = cleanedText
wikihowAll['cleanedSummary'] = cleanedSummary

wikihowAll.replace('', np.nan, inplace=True)
wikihowAll.dropna(axis=0, inplace=True)

cleanedText = np.array(wikihowAll['cleanedText'])
cleanedSummary = np.array(wikihowAll['cleanedSummary'])

textLength = 100
summaryLength = 10

shortText = []
shortSummary = []

for i in range(len(cleanedText)):
    if (len(cleanedSummary[i].split()) <= summaryLength and len(cleanedText[i].split()) <= textLength):
        shortText.append(cleanedText[i])
        shortSummary.append(cleanedSummary[i])

df = pd.DataFrame({'text': shortText, 'summary': shortSummary})

df['summary'] = df['summary'].apply(lambda x: '<BOS> ' + x + ' <EOS>')

xTrain, xTest, yTrain, yTest = train_test_split(np.array(df['text']), np.array(df['summary']), test_size=0.1,
                                                shuffle=True)

xToken = Tokenizer()
xToken.fit_on_texts(list(xTrain))

thresh = 4
cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in xToken.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if (value < thresh):
        cnt = cnt + 1
        freq = freq + value

xToken = Tokenizer(num_words=tot_cnt - cnt)
xToken.fit_on_texts(list(xTrain))
xTrainSeq = xToken.texts_to_sequences(xTrain)
xTestSeq = xToken.texts_to_sequences(xTest)
xTrain = pad_sequences(xTrainSeq, maxlen=textLength, padding='post')
xTest = pad_sequences(xTrainSeq, maxlen=textLength, padding='post')
xVoc = xToken.num_words + 1

yToken = Tokenizer()
yToken.fit_on_texts(list(yTrain))

thresh = 3
cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in yToken.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if (value < thresh):
        cnt = cnt + 1
        freq = freq + value

yToken = Tokenizer(num_words=tot_cnt - cnt)
yToken.fit_on_texts(list(yTrain))
yTrainSeq = yToken.texts_to_sequences(yTrain)
yTestSeq = yToken.texts_to_sequences(yTest)
yTrain = pad_sequences(yTrainSeq, maxlen=summaryLength, padding='post')
yTest = pad_sequences(yTestSeq, maxlen=summaryLength, padding='post')
yVoc = yToken.num_words + 1

ind = []
for i in range(len(yTrain)):
    cnt = 0
    for j in yTrain[i]:
        if j != 0:
            cnt = cnt + 1
    if (cnt == 2):
        ind.append(i)

yTrain = np.delete(yTrain, ind, axis=0)
xTrain = np.delete(xTrain, ind, axis=0)

ind = []
for i in range(len(yTest)):
    cnt = 0
    for j in yTest[i]:
        if j != 0:
            cnt = cnt + 1
    if (cnt == 2):
        ind.append(i)

yTest = np.delete(yTest, ind, axis=0)
xTest = np.delete(xTest, ind, axis=0)

latentDims = 400
embeddingDims = 300

embeddingsIndex = {}
f = open(".//glove.6B//glove.6B.300d.txt", encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddingsIndex[word] = coefs
f.close()

encoderEmbeddingMatrix = np.zeros(((len(xToken.word_index) + 1, embeddingDims)))
for word, i in xToken.word_index.items():
    embeddingVector = embeddingsIndex.get(word)
    if embeddingVector is not None:
        encoderEmbeddingMatrix[i] = embeddingVector

decoderEmbeddingMatrix = np.zeros(((len(yToken.word_index) + 1, embeddingDims)))
for word, i in yToken.word_index.items():
    embeddingVector = embeddingsIndex.get(word)
    if embeddingVector is not None:
        decoderEmbeddingMatrix[i] = embeddingVector

encoderInputs = Input(shape=(textLength,))
encoderEmbeddings = Embedding(len(xToken.word_index) + 1, embeddingDims, weights=[encoderEmbeddingMatrix],
                              input_length=textLength, trainable=False)(encoderInputs)
encoderLSTM1 = LSTM(latentDims, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2)
encoderOutput1, stateH1, stateC1 = encoderLSTM1(encoderEmbeddings)
encoderLSTM2 = LSTM(latentDims, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2)
encoderOutput2, stateH2, stateC2 = encoderLSTM2(encoderOutput1)
encoderLSTM3 = LSTM(latentDims, return_state=True, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
encoderOutputs, stateH, stateC = encoderLSTM3(encoderOutput2)

decoderInputs = Input(shape=(None,))
decoderEmbeddingLayer = Embedding(len(yToken.word_index) + 1, embeddingDims, weights=[decoderEmbeddingMatrix],
                                  trainable=False)
decoderEmbeddings = decoderEmbeddingLayer(decoderInputs)
decoderLSTM = LSTM(latentDims, return_sequences=True, return_state=True, dropout=0.1, recurrent_dropout=0.1)
decoderOutputs, decoderFWD, decoderBack = decoderLSTM(decoderEmbeddings, initial_state=[stateH, stateC])

attentionLayer = AttentionLayer(name='attention_layer')
attentionOutputs, attentionStates = attentionLayer([encoderOutputs, decoderOutputs])

decoderConcatInputs = Concatenate(axis=-1, name='concat_layer')([decoderOutputs, attentionOutputs])

decoderDense = TimeDistributed(Dense(yVoc, activation='softmax'))
decoderOutputs = decoderDense(decoderConcatInputs)

model = Model([encoderInputs, decoderInputs], decoderOutputs)

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

history = model.fit([xTrain, yTrain[:, :-1]], yTrain.reshape(yTrain.shape[0], yTrain.shape[1], 1)[:, 1:], epochs=50,
                    callbacks=[es], batch_size=500,
                    validation_data=([xTest, yTest[:, :-1]], yTest.reshape(yTest.shape[0], yTest.shape[1], 1)[:, 1:]))

reverseTargetWordIndex = yToken.index_word
reverseSourceWordIndex = xToken.index_word
targetWordIndex = yToken.word_index

encoderModel = Model(inputs=encoderInputs, outputs=[encoderOutputs, stateH, stateC])

decoderStateInputH = Input(shape=(latentDims,))
decoderStateInputC = Input(shape=(latentDims,))
decoderHiddenStateInput = Input(shape=(textLength, latentDims))

decoderEmbeddings2 = decoderEmbeddingLayer(decoderInputs)

decoderOutputs2, stateH2, stateC2 = decoderLSTM(decoderEmbeddings2,
                                                initial_state=[decoderStateInputH, decoderStateInputC])

attentionOutputInf, attentionStatesInf = attentionLayer([decoderHiddenStateInput, decoderOutputs2])
decoderInfConcat = Concatenate(axis=-1, name='concat')([decoderOutputs2, attentionOutputInf])

decoderOutputs2 = decoderDense(decoderInfConcat)

decoderModel = Model(
    [decoderInputs] + [decoderHiddenStateInput, decoderStateInputH, decoderStateInputC],
    [decoderOutputs2] + [stateH2, stateC2])


def decode_sequence(inputSeq):
    eOut, eH, eC = encoderModel.predict(inputSeq)
    targetSeq = np.zeros((1, 1))
    targetSeq[0, 0] = targetWordIndex['<BOS>']
    stop = False
    decoded_sentence = ''
    while not stop:
        outputTokens, h, c = decoderModel.predict([targetSeq] + [eOut, eH, eC])
        sampledTokenIndex = np.argmax(outputTokens[0, -1, :])
        sampledToken = reverseTargetWordIndex[sampledTokenIndex]
        if (sampledToken != '<EOS>'):
            decoded_sentence += ' ' + sampledToken
        if (sampledToken == '<EOS>' or len(decoded_sentence.split()) >= (summaryLength - 1)):
            stop = True
        targetSeq = np.zeros((1, 1))
        targetSeq[0, 0] = sampledTokenIndex
        eH, eC = h, c

    return decoded_sentence


def seq2summary(inputSeq):
    newString = ''
    for i in inputSeq:
        if ((i != 0 and i != targetWordIndex['<BOS>']) and i != targetWordIndex['<EOS>']):
            newString = newString + reverseTargetWordIndex[i] + ' '
    return newString


def seq2text(inputSeq):
    newString = ''
    for i in inputSeq:
        if (i != 0):
            newString = newString + reverseSourceWordIndex[i] + ' '
    return newString


O = []
P = []

for i in range(len(xTest)):
    try:
        o = seq2summary(yTest[i])
        p = decode_sequence(xTest[i].reshape(1, textLength))
        O.append(str(o))
        P.append(str(p))
    except Exception as e:
        pass

rouge = Rouge()
scores = rouge.get_scores(P, O, avg=True)

f = open("GloVeOutputScores.txt", "w")
f.write(scores)
f.close()

f = open("GloVeOutputExamples.txt", "w")

for i in range(30):
    f.write("Review:", seq2text(xTest[i]))
    f.write("Original summary:", seq2summary(yTest[i]))
    f.write("Predicted summary:", decode_sequence(xTest[i].reshape(1, textLength)))
    f.write("----------")

f.close()
