import numpy as np
from pprint import pprint
from unidecode import unidecode
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import json, os
from keras.layers import Embedding

import reader

GLOVE_DIR = "/home/gaurav/Downloads/dataset/embeddings/glove.6B"
EMBEDDING_DIM = 200

x, y , ac_num, ac_words = reader.get_data()


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

x_embedding = []
for node in x:
	node_embedding = []
	for sent in node:
		temp_sent = sent.split(" ")
		sent_embedding = sentence_embedding([temp_value.strip() for temp_value in temp_sent],"GLOVE")
		node_embedding.append(sent_embedding)
	x_embedding.append(node_embedding)




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
