import numpy as np
from pprint import pprint
from unidecode import unidecode
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json, os
from keras.layers import Embedding

import reader
from ptrnets.seq2seq import cells


from ptrnets.seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq, Pointer
# from keras.utils.test_utils import keras_test
from pprint import pprint
import keras.backend as K
import numpy as np
from ptrnets import utils


GLOVE_DIR = "/home/gaurav/Downloads/dataset/embeddings/glove.6B"
EMBEDDING_DIM = 200



embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


def sentence_embedding(sentence, embedding_type):
	'''
		sentence = a list of words ["The","fastest","person","on","the","planet"]
		embedding_type = GLOVE, Word2Vec
	'''
	if embedding_type == "GLOVE":
		sen_embedding = np.zeros(200,dtype=np.float32)
		for word in sentence:
			try:
				embedding = embeddings_index[word]
			except:
				embedding = np.zeros(200,dtype=np.float32)
			sen_embedding = sen_embedding + embedding
		return sen_embedding
	return None	

x, y , ac_num, ac_words = reader.get_data()
x_embedding = []
for node in x:
	node_embedding = []
	for sent in node:
		temp_sent = sent.split(" ")
		sent_embedding = sentence_embedding([temp_value.strip() for temp_value in temp_sent],"GLOVE")
		node_embedding.append(sent_embedding)
	x_embedding.append(node_embedding)
x_embedding = np.asarray(x_embedding)

#done with the data preperation step 

x_train,x_test = x_embedding[:int(x_embedding.shape[0]*.80)],x_embedding[int(x_embedding.shape[0]*.80):]
y_train,y_test = y[:int(y.shape[0]*.80)],y[int(y.shape[0]*.80):]

models = Pointer(output_dim=10, hidden_dim=200, output_length=10, input_shape=(10, 200), batch_size=1,bidirectional=False)
models.compile(loss='mse', optimizer='sgd',metrics = ['accuracy'])
print models.summary()
models.fit(x_train, y_train, epochs=10,batch_size=1)
print "Done fitting model"





# x = []
# for node in x_data:
# 	temp = []
# 	for sent in node:
# 		temp.append([sent])
# 	x.append(temp)

# text = ''
# x_new = " ".join(["".join(node) for node in x])
# # text = " ".join(x_new)
# print x_new
	

# tokenizer = Tokenizer()