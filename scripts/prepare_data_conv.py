import torch as th

from utils import make_long_tensor
from utils import make_float_tensor

PADDING = 0

# Convolution data preparing
def make_vocab_char(data):
	char_to_ix = {}
	char_to_ix["<padding>"] = 0
	for l, line in data:
		for c in line:
			if not c in char_to_ix:
				char_to_ix[c] = len(char_to_ix)
	return char_to_ix

# On convertit les char en index et on rajoute du padding
def line_to_char_ix(data, char_to_ix):
	res = []
	for l, line in data:
		tmp = [char_to_ix[c] for c in line]
		if len(tmp) < 140:
			tmp += [PADDING] * (140 - len(tmp))
		if len(tmp) == 140:
			res.append((l, tmp))
	return res

# Convertit une phrase de char en index selon le vocab char_to_ix
def text_to_char_ix(text, char_to_ix):
	res = [char_to_ix[c] for c in text]
	if len(res) < 140:
		res += [PADDING] * (140 - len(res))
	return res

# Convertit line char en tensor pytorch
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
