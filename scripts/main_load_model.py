#!/home/samuel/anaconda3/bin/python
import pickle

import torch as th
import torch.autograd as ag

from read_twit import make_data
from read_twit import open_twit
import prepare_data

import model

NB_TRAIN = 0
NB_DEV = 0
NB_TEST = 10000
NB_TWIT = NB_TEST + NB_DEV +  NB_TRAIN

use_cuda = False

print("Load data (%s tweets)..." % (NB_TWIT))

all_lines = open_twit("./res/Sentiment Analysis Dataset.csv")
data = make_data(all_lines, NB_TWIT)
char_to_ix = prepare_data.make_vocab_char(data)
all_data = prepare_data.line_to_char_ix(data, char_to_ix)
all_data = [x for x in all_data if len(x[1]) > 0]
data_test = all_data

def eval_model(model, dataset):
	model.eval()
	nbErr = 0
	total = 0
	nbPos = 0
	for y, x in dataset:
		x = prepare_data.make_long_tensor(x, use_cuda).view((1,-1))
		x = ag.Variable(x)
		out = model(x)
		out = out > 0.5
		out = out.view((1))
		total += 1
		if out.data[0] != int(y):
			nbErr += 1
	return nbErr, total

print("Load model...")
(model, _, _) = pickle.load(open("./model/1/model.p", "rb" ))

print("Test model...")
err, total = eval_model(model, data_test)
print("Test", err, "/", total)