WHAGLOVE.py is the main program file which contains all the codes for the model, preprocessing and training and will also generate the scores and results to .txt files.

sampleScoreOutput.txt contains the results of the latest run, the sore for the test data is present and also 20 sample examples have been attached for reference.

As Keras does not have inbuilt attention layer, a custom attention layer was used, source: https://github.com/thushv89/attention_keras/blob/master/src/layers/attention.py

wikihow DATASET: https://github.com/mahnazkoupaee/WikiHow-Dataset, the wikiHowAll dataset was used from the experiments and should be in the same folder as the other files, it can be downloaded from the given link.

pretrained GLOVE EMBEDDINGS: https://nlp.stanford.edu/projects/glove/, the pretrained glove embeddings used were Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip, and should be in the same folder as the other files, it can be downloaded from the given link.
