import torch as th

from utils import make_long_tensor
from utils import make_float_tensor

PADDING = 0
LIMIT_COUNT = 1

# Simple model data preparing
def make_vocab(data):
	word_to_ix = {}
	word_count = {}
	word_to_ix["<padding>"] = PADDING
	for l, line in data:
		for w in line:
			if not w in word_to_ix:
				word_to_ix[w] = len(word_to_ix)
				word_count[w] = 1
			word_count[w] += 1
	return word_to_ix, word_count

def line_to_ix(data, word_to_ix, word_count):
	res = []
	for l, line in data:
		line = [word_to_ix[w] for w in line if word_count[w] > LIMIT_COUNT]
		if len(line) > 0:
			res.append((l, line))
	return res

def make_tensor_list(data, batch_size):
	line = []
	labels = []
	while len(data) != 0:
		b_s = batch_size if len(data) >= batch_size else len(data)
		max_size = 0
		batch_line = []
		batch_label = []
		for i in range(b_s):
			l, sent = data.pop()
			max_size = len(sent) if max_size < len(sent) else max_size
			batch_line.append(sent)
			batch_label.append([int(l)])
		tmp = []
		for l in batch_line:
			l = l + [0] * (max_size - len(l))
			tmp.append(l)
		line.append(th.LongTensor(tmp))
		labels.append(th.LongTensor(batch_label))
	return line, labels

def make_tensor_list_offset(data, batch_size, use_cuda):
	line = []
	offsets = []
	labels = []
	while len(data) != 0:
		b_s = batch_size if len(data) >= batch_size else len(data)
		curr_offset = 0
		batch_line = []
		off_line = []
		batch_label = []
		for i in range(b_s):
			l, sent = data.pop()
			batch_line += sent
			off_line.append(curr_offset)
			curr_offset += len(sent)
			batch_label.append([int(l)])
		line.append(make_long_tensor(batch_line, use_cuda))
		offsets.append(make_long_tensor(off_line, use_cuda))
		labels.append(make_float_tensor(batch_label, use_cuda))
	return line, offsets, labels
