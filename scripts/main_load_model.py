#!/usr/bin/env python
import pickle

import torch as th
import torch.autograd as ag

from read_twit import make_data_conv
from read_twit import open_twit
import prepare_data_conv
import utils

import model

NB_TRAIN = 0
NB_DEV = 0
NB_TEST = 10000
NB_TWIT = NB_TEST + NB_DEV +  NB_TRAIN

use_cuda = False

print("Load model...")
(model, _, _, char_to_ix) = pickle.load(open("./model/1/model.p", "rb" ))

print("Load data (%s tweets)..." % (NB_TWIT))

all_lines = open_twit("./res/Sentiment Analysis Dataset.csv")
data = make_data_conv(all_lines, NB_TWIT)
all_data = prepare_data_conv.line_to_char_ix(data, char_to_ix)
all_data = [x for x in all_data if len(x[1]) > 0]
data_test = all_data

def eval_model(model, dataset):
	model.eval()
	nbErr = 0
	total = 0
	nbPos = 0
	answer = { 0:0, 1:0 }
	for y, x in dataset:
		x = utils.make_long_tensor(x, use_cuda).view((1,-1))
		x = ag.Variable(x)
		out = model(x)
		out = out > 0.5
		out = out.view((1))
		total += 1
		if out.item() != int(y):
			nbErr += 1
			answer[int(y)] += 1
	return nbErr, total, answer

print("Test model...")
err, total, answer = eval_model(model, data_test)
print("Test", err, "/", total)
print(answer)
