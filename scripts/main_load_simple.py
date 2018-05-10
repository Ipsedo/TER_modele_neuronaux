#!/usr/bin/env python
import pickle

import torch as th
import torch.autograd as ag

from read_twit import make_data
from read_twit import open_twit
import prepare_data
import utils

import model

NB_TRAIN = 0
NB_DEV = 0
NB_TEST = 10000
NB_TWIT = NB_TEST + NB_DEV +  NB_TRAIN

use_cuda = False

print("Load model...")
(model, _, _, words_to_ix, word_count) = pickle.load(open("./ModelEmbeddingBag.p", "rb" ))

print("Load data (%s tweets)..." % (NB_TWIT))

all_lines = open_twit("./res/Sentiment Analysis Dataset.csv")
data = make_data(all_lines, NB_TWIT)
all_data = prepare_data.line_to_ix(data, words_to_ix, word_count)
all_data = [x for x in all_data if len(x[1]) > 0]
data_test = all_data

def eval_model(model, test_data):
    model.eval()
    nbErr = 0
    total = 0
    answer = { 0:0, 1:0 }
    tot = {0:0,1:0}
    for y, x in test_data:
        x = utils.make_long_tensor(x, use_cuda)
        off = utils.make_long_tensor([0], use_cuda)
        x = ag.Variable(x)
        off = ag.Variable(off)
        out = model((x, off)) > 0.5
        out = out.view(1)
        total += 1
        tot[int(y)] += 1
        if out.item() != int(y):
            answer[int(y)] += 1
            nbErr += 1
    return nbErr, total, answer, tot

print("Test model...")
err, total, answer, tot = eval_model(model, data_test)
print("Test", err, "/", total)
print(answer)
print(tot)
