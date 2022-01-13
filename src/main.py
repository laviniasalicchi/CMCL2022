import time
from tool.tool_logger import *
from data.data_util import DataLoader, FeatureExtraction
from model.model_regression import train, predict
from evaluate.evaluate_metrix import evaluate
import pickle
import traceback
import sys

#MAPPERS = ['plsr', 'mlp', 'rf', 'lr', 'rr', 'svr', 'brr', 'elast', 'lgb'] #, 'gbr']
MAPPERS = ['lgb']
#USE_INTERACTIONS = [False, True]
USE_INTERACTIONS = [False]

#LABEL_HEADERS = ['nFix', 'FFD', 'GPT', 'TRT', 'fixProp']
LABEL_HEADERS = ['FFDAvg','FFDStd',	'TRTAvg',	'TRTStd']

LOG_FP = '../log/' +  \
			 time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + \
			 '.txt'

logger.setlogger(None, LOG_FP)

def gene_feature():
	# dump features
	trainset_name = '../new_data/train.csv'
	devset_name = '../new_data/dev.csv'
	#testset_name = '../data/CMCL_data/test_data.csv'
	# test_name = '../data/CMCL_data/test.csv'

	trainrawtext, trainlabels = DataLoader.load_dataset(trainset_name)
	devrawtext, devlabels = DataLoader.load_dataset(devset_name)
	# testtext, testlabels = DataLoader.load_dataset(testset_name)
	
	traintext, trainsent = DataLoader.merge_sent(trainrawtext)
	devtext, devsent = DataLoader.merge_sent(devrawtext)

	trainfeature = FeatureExtraction.feature_extract(traintext, trainsent)
	devfeature = FeatureExtraction.feature_extract(devtext, devsent)
	# testfeature = FeatureExtraction.feature_extract(testtext)
	#
	# fpfeature = open('feature.pkl', 'wb')
	# pickle.dump([trainfeature, devfeature, trainlabels, devlabels], fpfeature)

	# load dump features
	# fpfeature = open('feature.pkl', 'rb')
	# [trainfeature, devfeature, trainlabels, devlabels] = pickle.load(fpfeature)

	return trainfeature, devfeature, trainlabels, devlabels

def run_oneexp(MAPPER, USE_INTERACTION):
	trainfeature, devfeature, trainlabels, devlabels = gene_feature()

	for i_label in range(4):
		logging.info('$$$$$ regression feature, MAPPER, USE_INTERACTION: {}, {}, {}'
					 .format(LABEL_HEADERS[i_label], MAPPER, USE_INTERACTION))

		# training
		model = train(trainfeature, trainlabels[:, i_label], MAPPER, USE_INTERACTION)

		# prediction
		predictlabel = predict(devfeature, model, USE_INTERACTION)

		# evaluation
		evaluate(predictlabel, devlabels[:, i_label])

def main():
	# loop for mapper
	for MAPPER in MAPPERS:
		# loop for whether using interaction
		for USE_INTERACTION in USE_INTERACTIONS:
			logging.info('$$$$$ regression feature: {}, {}'.format(MAPPER, USE_INTERACTION))
			run_oneexp(MAPPER, USE_INTERACTION)
			logging.info('##### one experiment loop finished\n')

main()
