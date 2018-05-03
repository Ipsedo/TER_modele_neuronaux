#!/home/samuel/anaconda3/bin/python
import pickle

import torch as th
import torch.autograd as ag

from read_twit import tokenize_line_2
import prepare_data

class_to_sentiment = { 1 : "positif", 0 : "negatif" }

print("Load model...")
(model, _, _, vocab) = pickle.load(open("./model/1/model.p", "rb" ))

def read_input():
    text = "unsued" * 24
    while len(text) > 140:
        text = input("Enter a sentence (max 140 char) : ")
    return text

def input_to_variable(text, vocab):
    text = tokenize_line_2(text)
    idxs = prepare_data.text_to_char_ix(text, vocab)
    tensor = prepare_data.make_long_tensor(idxs, False)
    return ag.Variable(tensor.view(1, -1))

def test_user():
    text = read_input()
    x = input_to_variable(text, vocab)
    print("Model is predicting...")
    out = model(x) > 0.5
    pred = out.view(1).data[0]
    print("Model thinks that it is %s" % (class_to_sentiment[pred]))

NB_TEST = 3
for _ in range(NB_TEST):
    test_user()
