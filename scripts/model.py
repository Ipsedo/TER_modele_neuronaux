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
		out = out.view((-1, self.embeds_dim, self.sent_length))
		out = self.conv1D_1(out)
		out = F.relu(out)
		out = self.maxPool1D(out).view(-1, self.embeds_dim)
		out = self.linear1(out)
		return self.sig1(out)
