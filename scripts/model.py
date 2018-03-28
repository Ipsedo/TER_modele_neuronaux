import torch.nn as nn

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