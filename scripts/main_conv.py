#!/home/samuel/anaconda3/bin/python
import pickle

import torch as th
import torch.nn as nn
import torch.autograd as ag

from read_twit import make_data_conv
from read_twit import open_twit
import prepare_data

import model

# Dataset : TEST DEV TRAIN
NB_TRAIN = 500000
NB_DEV = 10000
NB_TEST = 10000
NB_TWIT = NB_TEST + NB_DEV +  NB_TRAIN

use_cuda = th.cuda.is_available()

print("Load data (%s tweets)..." % (NB_TWIT))

all_lines = open_twit("./res/Sentiment Analysis Dataset.csv")
data = make_data_conv(all_lines, NB_TWIT)

print(data[0])
print(data[6])
print(data[7])
print(data[15])
print(data[60])
print(data[75])
print(data[153])
print(data[179])
print(data[32])
print(data[32])

char_to_ix = prepare_data.make_vocab_char(data)
all_data = prepare_data.line_to_char_ix(data, char_to_ix)

all_data = [x for x in all_data if len(x[1]) > 0]

# print(all_data[0])

data = all_data[NB_TEST + NB_DEV:]
data_test = all_data[:NB_TEST]
data_dev = all_data[NB_TEST:NB_TEST + NB_DEV]

line, labels = prepare_data.line_char_to_tensor(data, 150, use_cuda)
# print(len(line), len(labels))
# print(type(line[0]), type(labels[0]))
# print(line[0].size(), labels[0].size())

print("Prepare model...")

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

EPOCH = 20
vocab_size = len(char_to_ix)
embedding_dim = 100

model = model.ConvModel2(vocab_size, embedding_dim, 0, 140)
learning_rate = 1e-3
loss_fn = nn.BCELoss()

if use_cuda:
    model.cuda()
    loss_fn.cuda()

optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

print("Learn model...")

for i in range(EPOCH):
	model.train()
	total_loss = 0
	for x, y in zip(line, labels):
		model.zero_grad()
		x = ag.Variable(x)
		y = ag.Variable(y)

		out = model(x)
		loss = loss_fn(out, y)
		total_loss += loss.data[0]
		loss.backward()
		optimizer.step()
	print("Epoch", i)
	"""if i % 10 == 0:
		err, total = eval_model(model, data_dev)
		print("Epoch (", i, ") : ", total_loss, ", test (err/total) : ", err, " / ", total, sep="")"""
err, total = eval_model(model, data_test)
print("Test", err, "/", total)

model.cpu()
loss_fn.cpu()
tosave = (model, optimizer, loss_fn)
pickle.dump( tosave, open( "model.p", "wb" ) )
