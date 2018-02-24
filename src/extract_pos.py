import nltk.data
import sys, pprint, argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from itertools import chain


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def thread_pos_tagging(thread_idx, dataset):
	X = list()
	#for data in dataset:
	for data in tqdm(dataset, position=thread_idx):
		# From an article to sentences.
		result = tokenizer.tokenize(data)
		tab_index = result[0].find('\t')
		result[0] = result[0][tab_index+1:] # skip PMID\t
		# Processing POS-Tagging
		for element in result:
			# first extract single word
			tmp = nltk.word_tokenize(element)
			# apply nltk upenn pos tag tree
			pos = nltk.pos_tag(tmp)
			X.append(pos)
	return X

def extract_pos(args):
	with open(args.sentences, 'r') as f, open(args.found, 'w') as fw:
		# Acquire all sentences
		raw_data = f.read().splitlines()[1:] # skip first line
		# Threading settings
		n_cores = multiprocessing.cpu_count()
		n_threads = n_cores * 2 if args.thread == None else args.thread
		print("Number of CPU cores: {:d}".format(n_cores))
		print("Number of Threading: {:d}".format(n_threads))
		# Slice data for each thread
		print(" - Slicing data for threading...")
		per_slice = int(len(raw_data) / n_threads)
		thread_data = list()
		# split dataset
		for itr in range(n_threads):
			# Generate indices for each slice
			idx_begin = itr * per_slice
			# last slice may be larger or smaller
			idx_end = (itr + 1) * per_slice if itr != n_threads - 1 else None
			#
			thread_data.append((itr, raw_data[idx_begin:idx_end]))
	
		print(" - Begin threading...")
		# Threading
		with Pool(processes=n_threads) as p:
			result = p.starmap(thread_pos_tagging, thread_data)
		print("\n\nAll threads completed.")
	
		for line in tqdm(list(chain.from_iterable(result))):
			fw.write('\n'.join('{} {}'.format(x[1],x[0]) for x in line))
			fw.write('\n#########split_sentence#########\n')
		print("File saved in {:s}".format(args.found))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("sentences", help="Input sentences to be recognized.")
	parser.add_argument("found", help="Sentences with key words")
	parser.add_argument("-t", "--thread", type=int, help="Number of threads \
						to run, default: 2 * number_of_cores") 

	args = parser.parse_args()

	extract_pos(args)