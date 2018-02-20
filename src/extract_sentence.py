import nltk.data
import sys, pprint
# import numpy as np

filename = sys.argv[1]

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
with open(filename, 'r') as f:
	dataset = f.read().splitlines()[1:] # skip first line
	X = list()
	count = 0
	print("start tokenizing...")
	for data in dataset:
		if count % 10 == 0:
			print("step {:3d} / {:3d}".format(count, len(dataset)))
		result = tokenizer.tokenize(data)
		tab_index = result[0].find('\t')
		result[0] = result[0][tab_index+1:] # skip PMID\t
		#pprint.pprint(result)
		for element in result:
			X.append(element)
		count += 1

	save_name = filename[:-4] + '_sentence.txt'
	with open(save_name, 'w') as fw:
		for element in X:
			fw.write(element+'\n')

	print(len(X))

