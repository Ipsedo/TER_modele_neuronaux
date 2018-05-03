import torch as th

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
