import pandas as pd
import textstat, string
import numpy as np
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
import torch
from sklearn import preprocessing
from wordfreq import word_frequency
import pickle
import copy
import logging
import sys
#import spacy
#from spacy_syllables import SpacySyllables
from indicnlp.syllable import  syllabifier



from lm_scorer.models.auto import AutoLMScorer as LMScorer
scorer = LMScorer.from_pretrained("gpt2", device="cuda:0")
###scorer = LMScorer.from_pretrained("gpt2")

#from stanfordcorenlp import StanfordCoreNLP
#nlp = StanfordCoreNLP(r'D:\workspace\STANFORD_models\stanford-corenlp-full-2018-10-05', lang='en')

class DataLoader():

	@staticmethod
	def load_dataset(dsname, withhead=True, istest=False):
		if withhead:
			names = ['language',	'sentence_id', 'word_id',	'word',	'FFDAvg','FFDStd',	'TRTAvg',	'TRTStd']
			dataset = pd.read_csv(dsname)
		else:
			dataset = pd.read_csv(dsname, header=None)

		df = dataset
		df = df.astype({'word_id': int})
		sent_n = 0
		sent_arr = []

		word_n = 0
		word_arr = []
		for i, rows in df.iterrows():
			if i == 0:
				sent_id = df['sentence_id'][i]
				word_n = -1
				
			if df['sentence_id'][i] != sent_id:
				sent_n +=1
				sent_id = df['sentence_id'][i]
				word_n = 0
			else:
				word_n += 1
			
			df['word'][i] = df['word'][i].replace(" ","")

			sent_arr.append(sent_n)
			word_arr.append(word_n)

		
		df['sent_n'] = sent_arr
		df['word_n'] = word_arr
			
		if istest:
			rawtext = df.iloc[:,[8,9,3,0]].to_numpy()
			labels = None
		else:
			rawtext = df.iloc[:, [8,9,3,0]].to_numpy()
			labels = df.iloc[:, 4:8].to_numpy()
		
		return rawtext, labels

	@staticmethod
	def merge_sent(rawtext):
		cur_sent_len = 0
		rawtext = np.insert(rawtext, 0, values=0, axis = 1)
		

		sents = []
		sent = ""

		for i in range(len(rawtext) - 1, -1, -1):
			item = rawtext[i]
			sent_len = item[0]
			sent_id = item[1]
			wordseq_id = item[2]
			wordtext = item[3]
			
			if cur_sent_len < wordseq_id:
				cur_sent_len = wordseq_id
				rawtext[i, 0] = cur_sent_len

			if wordseq_id == 0:
				sent = wordtext + ' ' + sent
				#sents.insert(0, sent[:-6])
				sents.insert(0, sent)
				rawtext[i, 0] = cur_sent_len
				cur_sent_len = 0
				sent = ""
			else:
				sent = wordtext + ' ' + sent
				rawtext[i, 0] = cur_sent_len
		

		# mergedtext = copy.deepcopy(rawtext)
		
		return rawtext, sents

class FeatureExtraction():

	@staticmethod
	def extract_vocabulary(list_files):

		# returns vocabulary of the train and of the test file

		vocab = {}
		stop_words = stopwords.words('english')

		for f in list_files:

			with open(f, "r", encoding='utf-8') as fl:
				for line in fl:

					# if line.startswith("id") or len(line.split("\t")) != 5: continue

					line = line.strip()
					try:
						_, _, sent, tok, _ = line.split("\t")
					except:
						_, _, sent, tok = line.split("\t")
					tok = tok.lower()
					sent = sent.translate(str.maketrans('', '', string.punctuation))
					sent = sent.lower()

					for word in sent.split(" "):
						if word in stop_words: continue
						vocab[word] = 0

					vocab[tok] = 0

		return vocab

	@staticmethod
	def get_vec(word, embs, PRINT_WORD=False):

		# Returns a vector for word from vector matrix embs
		print("IN GET_VEC")

		if word in embs.keys():
			try:
				return embs['_matrix_'][embs[word], :]
			except:
				print('{} should have been there but something went wrong when loading it!'.format(word))
				return []
		else:
			if PRINT_WORD:
				print('{} not in the dataset.'.format(word))

			return []

	@staticmethod
	def load_embeddings(fname, ds_words):
		# Load the embeddings from fname file, only for the words in the variable ds_words

		emb = {}
		matrix = []
		dims = 0
		with open(fname, 'r', encoding='utf-8', errors="ignore") as f:
			for line in f:
				line = line.strip().split()
				if dims == 0:
					if len(line) == 2:
						continue
					else:
						dims = len(line) - 1

				word = line[0]

				if word not in ds_words: continue

				if word in ['', ' ', '\t', '\n']:
					print('Word {} has no value.'.format(word))
					continue
				try:
					vec = [float(x) for x in line[1:]]
					if len(vec) == dims:
						array = np.array(vec)
					else:
						continue
				except:
					continue
				emb[word] = len(emb)
				matrix.append(array)

		emb['_matrix_'] = np.array(matrix)  # normalize()

		return emb

	@staticmethod
	def feature_extract(mergedtext, sents):
		features = []

		logging.info('USING FEATURES: ' + \
					 'CUR_WORD_POS, ' + \
					 'CUR_WORD_LEN, ' + \
					 'PREV_WORD_LEN, ' + \
					 'CUR_WORD_LOGFREQ, ' + \
					 'PREV_WORD_LOGFREQ, ' + \
					 'IS_UPPER, ' + \
					 'IS_CAPITAL, ' + \
					 'SYLLABLE_COUNT, ' + \
					 'READABILITY, ' + \
					 'SURPRISAL SCORE, '
					 )

		
		for i in range(0, len(mergedtext)):
			# if i % 10 == 0:
			#	 print(i)

		
			item = mergedtext[i]


			feat = []

			sentlen = item[0]
			sent_id = item[1]
			wordseq_id = item[2]
			wordtext = item[3]
			wordtext = wordtext.replace('<EOS>','')
			wordlanguage = item[4]
			
			


			wordpos = wordseq_id
			#CUR_WORD_POS	
			if sentlen != 0:		
				feat.append(wordpos / sentlen)
			else:
				feat.append(0.0)

			wordlen = len(wordtext)
			#print(wordtext, wordlen)
			#CUR_WORD_LEN
			feat.append(wordlen)
			#PREV_WORD_LEN
			if wordseq_id != 0:
				feat.append(len(mergedtext[i-1][3]))
			else:
				feat.append(0.0)

			# append with wordfreq lib
			#CUR_WORD_LOGFREQ
			word_freq_lib = word_frequency(wordtext.lower(), wordlanguage)

			if word_freq_lib == 0.0:
				feat.append(0.0)
			else:
				feat.append(-np.log(word_freq_lib))
			
			#PREV_WORD_LOGFREQ
			if wordseq_id != 0:
				word_freq_lib = word_frequency(mergedtext[i-1][3].lower(), wordlanguage)
				if word_freq_lib == 0.0:
					feat.append(0.0)
				else:
					feat.append(-np.log(word_freq_lib))
			else:
				feat.append(0.0)

			# # append the number of tokens before or after the target
			# try:
			# 	 splittedSent = nlp.word_tokenize(sent)
			# 	 index_token = splittedSent.index(tok)
			# 	 tokens_before, tokens_after = splittedSent.index(tok) - 1, len(splittedSent) - splittedSent.index(tok)
			# 	 feat.append(tokens_before / len(splittedSent))
			# 	 feat.append(tokens_after / len(splittedSent))
			# except ValueError:
			# 	 feat.append(0.0)
			# 	 feat.append(0.0)
			# 	 index_token = -1

			# append features for capitalization
			# one if the string is all upper case
			# IS_UPPER
			if wordtext == wordtext.upper():
				feat.append(1.0)
			else:
				feat.append(0.0)

			# one if the initial is upper case
			# IS_CAPITAL
			if wordtext[0] == wordtext[0].upper():
				feat.append(1.0)
			else:
				feat.append(0.0)

			# append the first features: the syllable count and the word length of the target
			# SYLLABLE_COUNT	!!!!!
			if wordlanguage not in ["hi", "zh"]:
				textstat.set_lang(wordlanguage)
				syllab = textstat.syllable_count(wordtext)
			if wordlanguage == "zh":
				syllab = wordlen
			feat.append(syllab)		
			print(wordtext, textstat.syllable_count(wordtext))	
			#nlp.add_pipe("syllables", after="tagger", config={"lang": "en_US"})

			# # extract the embedding of the target token
			# tok_vector = get_vec(tok, embs)
			# sent_vector = np.zeros(300)
			# # sum the vectors of the token in the sentence before the target token (if not stop words)
			# for word in splittedSent:
			#	 word = word.lower()
			#
			#	 if word == tok:
			#		 continue
			#
			#	 if word in stop_words:
			#		 continue
			#	 else:
			#		 w_vec = get_vec(word, embs)
			#
			#		 if len(w_vec) != 0:
			#			 sent_vector = np.sum([sent_vector, w_vec], axis=0)
			#		 else:
			#			 continue
			#
			# is_all_zero = np.all((sent_vector == 0))

			# # if the vector of the target is in the dictionary, and if the sentence vector is != all zeroes, append the cosine between the two to the features
			# if is_all_zero or len(tok_vector) == 0:
			#	 feat.append(0.0)
			# else:
			#	 feat.append(cosine(sent_vector, tok_vector))

			# # append the feature of the prob score of the sentence according to gpt2.
			# # Notice that you can also extract the probability of the single token: the method scorer.sentence_score returns a tuple of three lists: ([probabilities, ids, tokens])
			# feat.append(scorer.sentence_score(sent, reduce="mean"))
			#
			

			splittedSent = sents[sent_id].split(' ')
			index_token = splittedSent.index(wordtext)

			'''
			if index_token != -1:
				feat.append(scorer.tokens_score(sents[sent_id])[0][index_token])
			else:
				feat.append(0.0)
			'''

			# # append the feature of a readability index for the sentence
			# READABILITY
			if wordlanguage not in ["hi", "zh"]:
				readb = textstat.flesch_reading_ease(sents[sent_id])

			feat.append(readb)


			# print (feat)
			features.append(feat)

		features_array = np.array(features, dtype='float64')

		# fpout = open('../output/features_array.txt', 'w', encoding='utf-8')
		# for item in features_array:
		#	 fpout.writelines(str(item) + '\n')
		# fpout.close()

		return features_array
