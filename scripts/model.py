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
		self.conv1D_1 = nn.Conv1d(embedding_size, embedding_size, 3, stride=1)
		self.sig1 = nn.Sigmoid()
		self.linear1 = nn.Linear(embedding_size * (self.sent_length - 2), 1)
		self.sig2 = nn.Sigmoid()

	def forward(self, inputs):
		out = self.embedding(inputs)
		#toAdd = 140 - out.size()[0]
		#out = F.pad(out, (toAdd/2, toAdd/2), "constant", 0)
		out = out.view((-1, self.embeds_dim, self.sent_length))
		#out = F.conv1d(out, )
		out = self.conv1D_1(out)
		out = F.relu(out)
		out = out.view(-1, self.embeds_dim * (self.sent_length - 2))
		#out = self.sig1(out)
		out = self.linear1(out)
		return self.sig2(out)
