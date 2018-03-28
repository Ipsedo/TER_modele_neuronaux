import torch as th
import torch.nn.functional as F

PADDING = 0
LIMIT_COUNT = 0

def make_long_tensor(list, use_cuda):
	if use_cuda:
		return th.cuda.LongTensor(list)
	else:
		return th.LongTensor(list)

def make_float_tensor(list, use_cuda):
	if use_cuda:
		return th.cuda.FloatTensor(list)
	else:
		return th.FloatTensor(list)

def make_vocab_char(data):
	char_to_ix = {}
	char_to_ix["<padding>"] = 0
	for l, line in data:
		for w in line:
			for c in w:
				if not c in char_to_ix:
					char_to_ix[c] = len(char_to_ix)
	return char_to_ix

def line_to_char_ix(data, char_to_ix):
	res = []
	for l, line in data:
		tmp = []
		for w in line:
			tmp += [char_to_ix[c] for c in w]
		if len(tmp) < 140:
			tmp += [PADDING] * (140 - len(tmp))
		res.append((l, tmp))
	return res

def line_char_to_tensor(data, batch_size, use_cuda):
	res_line = []
	res_label = []
	while len(data) != 0:
		b_s = batch_size if len(data) >= batch_size else len(data)
		batch_line = []
		batch_label = []
		for i in range(b_s):
			l, line = data.pop()
			batch_line.append(line)
			batch_label.append([int(l)])
		res_line.append(make_long_tensor(batch_line, use_cuda))
		res_label.append(make_float_tensor(batch_label, use_cuda))
	return res_line, res_label

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
