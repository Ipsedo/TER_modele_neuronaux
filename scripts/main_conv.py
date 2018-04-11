#!/home/samuel/anaconda3/bin/python

"""
m = nn.Conv1d(16, 33, 3, stride=1)
	stride=2
	33 -> taille "d'embedding de retour"
	3 -> fenêtre
input = torch.randn(20, 16, 50)
	20 batch
	16 taille embedding
	50 longueur phrase
output = m(input)
"""


import torch as th
import torch.nn as nn
import torch.autograd as ag

from read_twit import make_data
from read_twit import open_twit
import prepare_data

import model

NB_TWIT = 100000
LIMIT = int(NB_TWIT * 9 / 10)

use_cuda = th.cuda.is_available()

all_lines = open_twit("./res/Sentiment Analysis Dataset.csv")
data = make_data(all_lines, NB_TWIT)
#print(data[32])
char_to_ix = prepare_data.make_vocab_char(data)
all_data = prepare_data.line_to_char_ix(data, char_to_ix)

all_data = [x for x in all_data if len(x[1]) > 0]

# print(all_data[0])

data = all_data[:LIMIT]
data_test = all_data[LIMIT:]

line, labels = prepare_data.line_char_to_tensor(data, 150, use_cuda)
print(len(line), len(labels))
print(type(line[0]), type(labels[0]))
print(line[0].size(), labels[0].size())

def eval_model(model, test_data):
	model.eval()
	nbErr = 0
	total = 0
	nbPos = 0
	for y, x in test_data:
		x = prepare_data.make_long_tensor(x, use_cuda).view((1,-1))
		x = ag.Variable(x)
		out = model(x)
		out = out > 0.5
		out = out.view((1))
		total += 1
		if out.data[0] != int(y):
			nbErr += 1
	return nbErr, total

EPOCH = 300
vocab_size = len(char_to_ix)
embedding_dim = 100

model = model.ConvModel(vocab_size, embedding_dim, 0, 140)
learning_rate = 1e-3
loss_fn = nn.BCELoss()

if use_cuda:
    model.cuda()
    loss_fn.cuda()

optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

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
	if i % 10 == 0:
		err, total = eval_model(model, data_test)
		print("Epoch (", i, ") : ", total_loss, ", test (err/total) : ", err, " / ", total, sep="")
