from attention import AttentionLayer
import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer 
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.python.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping
import warnings
pd.set_option("display.max_colwidth", 200) # For jupyter notebook
warnings.filterwarnings("ignore")
import nltk
nltk.download('stopwords')
import tensorflow_hub as hub
from bert import tokenization
import bert
from tqdm import tqdm_notebook
from transformers import *
bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2" # pre-trained model for tensorflow 2
bert_layer = hub.KerasLayer(bert_path,trainable=False) # Pre-trained BERT layer for encoding
from rouge import Rouge 
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K 



data=pd.read_csv("./wikihowAll.csv") # This dataset is large. It takes up about 20 mins to load on peregrine

text_length = 100
summary_length = 12

"""#Drop empty rows"""

data.replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)

data.columns

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}
# Cleanup text and summaries
stop_words = set(stopwords.words('english')) 
def text_cleaner(text):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    tokens = [w for w in newString.split() if w not in stop_words] # Remove stopwords
    long_words=[]
    for i in tokens:
        if len(i)>=3:                  #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()

cleaned_text = []
for t in data['text']:
    cleaned_text.append(text_cleaner(t))

def summary_cleaner(text):
    newString = re.sub('"','', text)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    newString = newString.lower()
    tokens=newString.split()
    newString=''
    for i in tokens:
        if len(i)>1:                                 
            newString=newString+i+' '  
    return newString

cleaned_summary = []
for t in data['title']:
    cleaned_summary.append(summary_cleaner(t))

data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)

# Split dataset but make sure that each time it makes the same split
x_tr,x_val,y_tr,y_val=train_test_split(np.array(data['cleaned_text']),np.array(data['cleaned_summary']),test_size=0.1,random_state=0,shuffle=True)


# The following functions are needed to convert input to the format BERT requires. 
def _get_segments(sentences):
    sentences_segments = []
    for sent in sentences:
      temp = []
      i = 0
      for token in sent.split():
        temp.append(i)
        if token == "[SEP]":
          i += 1
      sentences_segments.append(temp)
      
    return sentences_segments

def _get_inputs(df,_maxlen,tokenizer):


    maxqnans = _maxlen
    pattern = '[^\w\s]+|\n' # remove everything including newline (|\n) other than words (\w) or spaces (\s)
    sentences = ["[CLS] " + " ".join(tokenizer.tokenize(text.replace(pattern, ''))[:maxqnans-2])+" [SEP]" for text in df]
    sentences = [s.strip() for s in sentences]

    #generate masks
    # bert requires a mask for the words which are padded. 
    # Say for example, maxlen is 100, sentence size is 90. then, [1]*90 + [0]*[100-90]
    sentences_mask = [[1]*len(sent.split())+[0]*(_maxlen - len(sent.split())) for sent in sentences]
 
    sentences_padded = [sent + " [PAD]"*(_maxlen-len(sent.split())) if len(sent.split())!=_maxlen else sent for sent in sentences ]
    sentences = [re.sub(' +', ' ',s) for s in sentences]
    sentences_converted = [tokenizer.convert_tokens_to_ids(s.split()) for s in sentences_padded]

  
    #generate segments
    # for each separation [SEP], a new segment is converted
    sentences_segment = _get_segments(sentences_padded)

    genLength = set([len(sent.split()) for sent in sentences_padded])

    if len(genLength)!=1: 
      print(genLength)
      raise Exception("sentences are not of same size")



    #convert list into tensor integer arrays and return it
    return [tf.cast(sentences_converted,tf.int32), tf.cast(sentences_segment,tf.int32), tf.cast(sentences_mask,tf.int32)]

vocab_file1 = bert_layer.resolved_object.vocab_file.asset_path.numpy()
bert_tokenizer_tfhub = bert.bert_tokenization.FullTokenizer(vocab_file1, do_lower_case=True)


# Convert inputs for feeding into encoder
bert_inputs = _get_inputs(df=list(x_tr),tokenizer=bert_tokenizer_tfhub,_maxlen=text_length)

# Vocab size for decoder
output_size = len(bert_tokenizer_tfhub.vocab.keys()) # to makesure the dimensions are same


# convert inputs for feeding into decoder
decoder_inputs = _get_inputs(df=list(y_tr),tokenizer=bert_tokenizer_tfhub,_maxlen=summary_length)

# convert val inputs for feeding into encoder
bert_inputs_test = _get_inputs(df=list(x_val),tokenizer=bert_tokenizer_tfhub,_maxlen=text_length)

# Convert val inputs for feeding into decoder
decoder_inputs_test = _get_inputs(df=list(y_val),tokenizer=bert_tokenizer_tfhub,_maxlen=summary_length)


# Removes any premade graph. Useful when using jupyter notebook
K.clear_session()

latent_dim = 500 # Hidded and embedding dims

# Encoder
input_word_ids = Input((text_length,), dtype=tf.int32, name='input_word_ids')
input_masks = Input((text_length,), dtype=tf.int32, name='input_masks')
input_segments = Input((text_length,), dtype=tf.int32, name='input_segments')
_, sout = bert_layer([input_word_ids, input_masks, input_segments]) # Bert layer is called to get the BERT embeddings


# #embedding layer

#encoder lstm 1
encoder_lstm1 = LSTM(latent_dim,return_sequences=True,return_state=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2)
encoder_output1, state_h1, state_c1 = encoder_lstm1(sout)

#encoder lstm 2
encoder_lstm2 = LSTM(latent_dim,return_sequences=True,return_state=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

#encoder lstm 3
encoder_lstm3=LSTM(latent_dim, return_state=True, return_sequences=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2)
encoder_outputs, state_h, state_c= encoder_lstm3(encoder_output2)

dec_input = Input(shape=(None,))
dec_emb_layer = Embedding(output_size+1, latent_dim,trainable=True)
dec_emb = dec_emb_layer(dec_input)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, activation='tanh', dropout=0.2, recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#dense layer
decoder_dense =  TimeDistributed(Dense(output_size+1, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model 
model = Model([input_word_ids, input_masks, input_segments, dec_input], decoder_outputs)

model.summary()

# Extract out arrays from the list of converted inputs
sen_x, sen_x_1, sen_x_2 = bert_inputs
sen_y, _, _ = decoder_inputs
sen_x_val, sen_x_val_1, sen_x_val_2 = bert_inputs_test
sen_y_val, _, _ = decoder_inputs_test


optimizer = tf.keras.optimizers.RMSprop()
# To automatically convert to one-hot-encoding
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# To load pre-trained model
model.load_weights('./models/bert_model_kindle_encoder_without_stopwords')

es = EarlyStopping(mode='min', verbose=1,patience=2, monitor='val_loss')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./models/bert_model_kindle_encoder_without_stopwords',
                                                 save_weights_only=True,
                                                 verbose=1)


history=model.fit([sen_x, sen_x_1, sen_x_2, sen_y[:, :-1]], tf.reshape(sen_y,(sen_y.shape[0], sen_y.shape[1],1))[:,1:], batch_size=128,epochs=100,callbacks=[es, cp_callback], validation_data = ([sen_x_val, sen_x_val_1, sen_x_val_2, sen_y_val[:, :-1]], tf.reshape(sen_y_val,(sen_y_val.shape[0], sen_y_val.shape[1],1))[:,1:]))



# Plot train/test loss during training. Using tensorboard would have also been a good idea
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.savefig('train_loss_with_stopwords')

#inference
# encoder inference
encoder_model = Model(inputs=[input_word_ids, input_masks, input_segments],outputs=[encoder_outputs, state_h, state_c])

# decoder inference
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(text_length,latent_dim))

dec_emb2 = dec_emb_layer(dec_input)
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
[dec_input] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
[decoder_outputs2] + [state_h2, state_c2])

def decode_sequence(sentence):
      
      output_text = '[CLS]'
      sentence = sentence.strip()
      inp, inp_1, inp_2 = _get_inputs(df=[sentence],tokenizer=bert_tokenizer_tfhub,_maxlen=text_length)
      e_out, e_h, e_c = encoder_model.predict([inp, inp_1, inp_2])
      target_seq = np.zeros((1,1))
      stop_condition = False
      target_seq[0,0] = bert_tokenizer_tfhub.convert_tokens_to_ids([output_text])[0]
      output_text = ''
      while not stop_condition:
          output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
          sampled_token_index = np.argmax(output_tokens[0, 0, :])
          sampled_token = bert_tokenizer_tfhub.convert_ids_to_tokens([sampled_token_index])
          if sampled_token[0] != "[SEP]":
            output_text += " "+sampled_token[0]

          if sampled_token[0] == '[SEP]' or len(output_text.split()) > (summary_length):
            stop_condition = True
          output_text = output_text.strip()
          target_seq[0,0] = sampled_token_index
          e_h, e_c = h, c


      return output_text

# Print predicted summaries on the entire test set
rouge = Rouge()
rouge_list = []
golden_list = []
predicted_list = []
for i in range(len(x_val)):
    print("Review:",(x_val[i]))
    print("Original summary:",(y_val[i]))
    golden_list.append(y_val[i])
    predicted_summary = decode_sequence(x_val[i]).replace(' ##', '') # This is needed as BERT does subword tokenization and we need to convert it back
    predicted_list.append(predicted_summary)
    print("Predicted summary:",predicted_summary)
    print("\n")

print('Average rogue: ', rouge.get_scores(predicted_list, golden_list, avg=True))