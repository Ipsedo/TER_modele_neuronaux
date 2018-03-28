#!/home/samuel/anaconda3/bin/python
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

words_to_ix, word_count = prepare_data.make_vocab(data)
#print(len(words_to_ix))

all_data = prepare_data.line_to_ix(data, words_to_ix, word_count)
all_data = [x for x in all_data if len(x[1]) > 0]
data = all_data[:LIMIT]
data_test = all_data[LIMIT:]
#print(data_test)
#print(data[32])

def eval_model(model, test_data):
	model.eval()
	nbErr = 0
	total = 0
	for y, x in test_data:
		x = prepare_data.make_long_tensor(x, use_cuda)
		off = prepare_data.make_long_tensor([0], use_cuda)
		x = ag.Variable(x)
		off = ag.Variable(off)
		out = model((x, off)) > 0.5
		out = out.view((1))
		total += 1
		if out.data[0] != int(y):
			nbErr += 1
	# On retourne le nombre d'erreur et le nombre d'exemples de test traités
	return nbErr, total

line, off, labels = prepare_data.make_tensor_list_offset(data, 50, use_cuda)
#print(len(line), len(off), len(labels))

EPOCH = 100
vocab_size = len(words_to_ix)
embedding_dim = 50

model = model.EmbeddingBagModel(vocab_size, embedding_dim)
learning_rate = 1e-1
loss_fn = nn.BCELoss()

if use_cuda:
    model.cuda()
    loss_fn.cuda()

optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(EPOCH):
	model.train()

	total_loss = 0

	for x, y, offset in zip(line, labels, off):
		model.zero_grad()
		x = ag.Variable(x)
		y = ag.Variable(y)
		offset = ag.Variable(offset)

		out = model((x, offset))
		loss = loss_fn(out, y)
		total_loss += loss.data[0]
		loss.backward()
		optimizer.step()
	if i % 10 == 0:
		model.eval()
		err, total = eval_model(model, data_test)
		print("Epoch (", i, ") : ", total_loss, ", test (err/total) : ", err, " / ", total, sep="")
