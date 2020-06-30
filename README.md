# Language Technology Project | University of Groningen

In this project we compare three summarization models on the WikiHow dataset.

Each of the model is uses a different embedding to encode the sentences. The results are compared and presented in the report.

Embeddings used are

- Word2Vec
- GloVe
- BERT

## Word2Vec

---

See word2vect_README.pdf for word2vec model

## BERT

---

The `bert_encoder_embedding.py` file contains the code to train and test the model. The outputs generated are stored in `without_stopwords_output.txt`. `attention.py` file was obtained from the internet as keras does not have an inbuilt attention layer.

### Test/Val Loss

![](Bert-model/without_stopwords.png)
