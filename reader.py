'''
	Author: geraltofrivia.
	This file reads the data and creates the input and output labels.
'''


import os
import sys
import xmltodict
from pprint import pprint

import numpy as np

# Some Macros
DATA_DIR = "data/"




file_names = [ x for x in os.listdir(DATA_DIR) if '.xml' in x ]

def resolve(key, edges):
	answer = None

	if key[0] == u'a':
		
		return u'e' + key[1:]

	for edge in edges:
		
		if key == edge[0]:
			#Then this triple has the desired key as a 'c' variable. Treat it's last value as the desired thing.
			answer = edge[2]

	if not answer[0] == 'e':
		answer = resolve(answer, edges)

	return answer

def get_data():
	# Empty lists
	X = []
	Y = []

	# For every filename, fetch it and do parsing voodoo
	for file_name in file_names:
		file_data = open( os.path.join(DATA_DIR, file_name) ).read()
		file_parsed = xmltodict.parse(file_data)

		edges = file_parsed['arggraph']['edge']
		texts = file_parsed['arggraph']['edu']

		'''
			Output labels
		'''
		edges = [ (edge['@id'], edge['@src'], edge['@trg']) for edge in edges ]


		y = np.zeros((len(texts), len(texts)))
		for edge in edges:
			if edge[1][0] != 'e':
				src = edge[1]
				trg = edge[2]

				# print src, trg

				src = resolve(src, edges)
				trg = resolve(trg, edges)

				# print "\t", src, trg

				y[int(src[1])-1][int(trg[1])-1] = 1.0

		Y.append(y)

		'''
			Input Labels
		'''
		x = [ '' for temp in texts ]
		for text in texts:
			x[int(text[u'@id'][1])-1] = text['#text']

			
		X.append(x)

	return X
	return np.asarray(Y)


if __name__ == "__main__":
	get_data()