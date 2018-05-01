import torch as th
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingBagModel(nn.Module):

	def __init__(self, vocab_size, embedding_size):
		super(EmbeddingBagModel, self).__init__()
		self.embeddings = nn.EmbeddingBag(vocab_size, embedding_size, mode='sum')
		self.linear1 = nn.Linear(embedding_size, 1)
		self.sig1 = nn.Sigmoid()
		#self.linear2 = nn.Linear(100, 1)
		#self.sig2 = nn.Sigmoid()

	def forward(self, inputs):
		(x, off) = inputs
		embeds = self.embeddings(x, off)
		out = self.linear1(embeds)
		out = self.sig1(out)
		#out = self.linear2(out)
		#out = self.sig2(out)
		return out

class ConvModel(nn.Module):

	def __init__(self, vocab_size, embedding_size, padding, sent_length):
		super(ConvModel, self).__init__()
		self.embeds_dim = embedding_size
		self.sent_length = sent_length
		self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding)
		self.conv1D_1 = nn.Conv1d(embedding_size, embedding_size, 7, stride=1)
		self.maxPool1D = nn.MaxPool1d(self.sent_length - 6)
		self.linear1 = nn.Linear(embedding_size, 1)
		self.sig1 = nn.Sigmoid()

	def forward(self, inputs):
		out = self.embedding(inputs)
		print("1 ", out.size())
		out = out.view((-1, self.embeds_dim, self.sent_length))
		print("2", out.size())
		out = self.conv1D_1(out)
		out = F.relu(out)
		out = self.maxPool1D(out).view(-1, self.embeds_dim)
		out = self.linear1(out)
		return self.sig1(out)

class ConvModel2(nn.Module):

	def __init__(self, vocab_size, embedding_size, padding, sent_length):
		super(ConvModel2, self).__init__()

		self.embeds_dim = embedding_size
		self.sent_length = sent_length
		self.in_channel_conv = 1
		self.out_channel_conv = 100

		self.embedding = nn.Embedding(
			vocab_size,
			embedding_size,
			padding_idx=padding)

		kernel_sizes = [3, 4, 5]

		self.conv = nn.ModuleList([ \
			nn.Conv2d( \
				self.in_channel_conv, \
				self.out_channel_conv, \
				(k, self.embeds_dim)) \
			for k in kernel_sizes])

		self.maxPool1D = nn.MaxPool1d(self.sent_length - 2, 100)

		self.maxpool = nn.ModuleList([ \
			nn.MaxPool1d(self.sent_length - k + 1, self.out_channel_conv) \
			for k in kernel_sizes])

		self.linear1 = nn.Linear(self.out_channel_conv * len(kernel_sizes), 1)
		self.dropout = nn.Dropout(2e-1)
		self.sig1 = nn.Sigmoid()

	def forward(self, inputs):
		# -1 pour taille de batch
		# embeds : (-1, self.sent_length, self.embeds_dim)
		# unsqueeze(1) : (-1, 1, self.sent_length, self.embeds_dim)
		out = self.embedding(inputs).unsqueeze(1)
		# conv : (-1, self.out_channel_conv, self.sent_length - k + 1, 1)
		# squeeze(3) : (-1, self.out_channel_conv, self.sent_length - k + 1)
		outs = [F.relu(conv2d(out)).squeeze(3) for conv2d in self.conv]
		# pool : (-1, self.out_channel_conv, 1)
		# squeeze(2) : (-1, self.out_channel_conv)
		outs = [pool(conv_out).squeeze(2) for conv_out, pool in zip(outs, self.maxpool)]
		# concaténation des différentes convolution et maxpool
		out = th.cat(outs, 1)
		out = self.linear1(out)
		out = self.dropout(out)
		return self.sig1(out)
