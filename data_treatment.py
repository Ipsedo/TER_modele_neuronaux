#!/home/samuel/anaconda3/bin/python
import re

all_data_train = open('./tasksv11/en/qa1_single-supporting-fact_train.txt', 'r')
all_data_train = all_data_train.read()
all_data_test = open("./tasksv11/en/qa1_single-supporting-fact_test.txt", "r")
all_data_test = all_data_test.read()

# https://github.com/keras-team/keras/blob/master/examples/babi_rnn.py
def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

split_line_train = all_data_train.split("\n")

words = re.sub("[0-9]+|\t|\n|\r", "", all_data_train+all_data_test)
print(type(words))
words = tokenize(words)

def make_vocab (l_words):
	words_to_ix = {}
	for w in l_words:
		if w not in words_to_ix:
			words_to_ix[w] = len(words_to_ix)
	return words_to_ix

words_to_ix = make_vocab(words)
print(words_to_ix)

def sentence_to_ix(words, vocab):
	res = []
	for w in words:
		res.append(vocab[w])
	return res

def format_line(line):
	splitted_line = tokenize(line)
	l = sentence_to_ix(splitted_line[1:len(splitted_line)], words_to_ix)
	return l

def format_answer(answer):
	splitted_answer = answer.split("\t")
	id_and_answer = splitted_answer[0]
	answer = sentence_to_ix([splitted_answer[1]], words_to_ix)[0]
	num_line_answer = int(splitted_answer[2])
	sent = format_line(id_and_answer)
	return sent, answer, num_line_answer

def make_data_train(splitted_line):
	res = []
	buff = []
	start = 1
	for line in splitted_line:
		if line == "":
			continue
		i = int(re.search('^[0-9]+', line).group(0))
		if i % 3 == 0:
			s, ans, idx = format_answer(line)
			res.append((buff, (s, ans, start)))
			start = i + 1
			buff = []
		else:
			line = format_line(line)
			buff.append((i, line))
		if i % 3 == 0:
			start += 1
		if i % 15 == 0:
			start = 1
	return res

def make_data_2(splitted_line):
	res = []
	for i in range(int(len(splitted_line) / 15)):
		sent = []
		quest_ans = []
		for j in range(5):
			line1 = splitted_line[i * 15 + j * 3]
			line2 = splitted_line[i * 15 + j * 3 + 1]
			answ = splitted_line[i * 15 + j * 3 + 2]
			quest, ans, idx = format_answer(answ)
			i1 = int(re.search('^[0-9]+', line1).group(0))
			i2 = int(re.search('^[0-9]+', line2).group(0))
			line1 = format_line(line1)
			line2 = format_line(line2)
			sent.append((i1, line1))
			sent.append((i2, line2))
			quest_ans.append((quest, ans, idx))
		res.append((sent, quest_ans))
	return res

data = make_data_2(split_line_train)
sentences, quest = data[12]
print("sentences")
print(sentences)
print("questions")
print(quest)

"""
	https://arxiv.org/pdf/1502.05698.pdf
	"They work by reading the story until the point they reach a question and hen have to output an answer"
		--> Pb si réponse est dans un substory précédent
	une story = 5 sous stories

	Idée 1 (dans réseau recurrent) :
	- On passe la 1ere sous story et la 1ere question -> on obtient la 1ere réponse
	- On passe la 1ere et la 2e sous story avec la 2e question -> on obtient la 2e réponse
	- etc... jusqu'à arriver à la dernière question
	---> A clarifier, 
		Qu'est-ce que l'on veut prédire ?
			- Un index de mot / phrase ?
			- La sortie (mot / phrase) qui a la plus grande valeur en fonction de la question
		Besoin de faire une distinction entre phrase et question
			- Question toujours la dernière de sa sous story,
			on ne récupère que le dernier output si on utilise un RNN par exemple

	Mais pb, il faut index phrases (et si possible le mot).
	- Sommer les embeddings des mots d'une phrase ?
	- Prédire l'index du mot dans la story ? (on incrémente les mots en concatenant toute les phrases)

	Prochain TP :
	- Adapter les données à l'idée 1 (si ça vaut le coup d'éssayer l'idée 1)
	- Passer les données sous pytorch (Tensor)
	- Voir si on utilise RNN ou LTSM ou MemNN
	- Tester sur un modèle simple (embedding + linear)
	- Finir de lire les pdf https://arxiv.org/pdf/1502.05698.pd


	Exemple :

		1 blalb
		2 sdsjd
		3 [question]
		--> On passe dans le RNN

		1 blalb
		2 dede
		4 deret
		5 kzjzi
		6 [question]
		--> On passe dans le RNN

"""