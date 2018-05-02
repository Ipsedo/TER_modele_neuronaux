
# yoon kim 2014

import re

def open_twit(filename):
	all_twit = open(filename, "r").read()
	all_line = all_twit.split("\n")
	description = all_line[0]
	# print(description)
	all_line = all_line[1:]
	return all_line

regex_balise = "&.+;"
regex_url = "http[a-z0-9\.\\\:]+"
#regex_happy_smiley = "(:|=|;|x)-*(\)|d|p)+"
#regex_sad_smiley = "(:|=|;|x)-*(\(|/|\||o|\$|\[)+"
regex_identifiant = "@[a-z0-9]+"
regex_not_word = "([^a-z ])+"
#regex_special_char = "(\.|-|\*|'|\")+"

def tokenize_line(line):
	#line =  "".join([w + " " for w in line.split(" ") if w != ""])
	line = line.lower()
	line = re.sub(regex_balise, "", line)
	line = re.sub(regex_url, "", line)
	#line = re.sub(regex_happy_smiley, "happy_smiley", line)
	#line = re.sub(regex_sad_smiley, "sad_smiley", line)
	line = re.sub(regex_identifiant, "", line)
	line = re.sub(regex_not_word, "", line)
	#line = re.sub(regex_special_char, "", line)
	return [w.strip(" ") for w in re.split('(\W+)', line) if w.strip(" ")]

def tokenize_line_2(line):
	#line =  "".join([w + " " for w in line.split(" ") if w != ""])
	line = line.lower()
	line = re.sub(regex_balise, "", line)
	line = re.sub(regex_url, "", line)
	line = re.sub(regex_not_word, "", line)
	return line

"""regex = "( |,|*|)"

def format_word(words):
	c_old = words[0]
	res = "" + c_old
	max_occurence = 1
	for c in words[1:]:
		if c_old != c:
			max_occurence = 1
		else:
			max_occurence += 1
		if max_occurence < 3:
			res += c
		c_old = c
	return res"""

def make_data(all_line, max_ex):
	res = []

	for i, line in enumerate(all_line):
		if line == "":
			continue
		splitted = line.split(",")
		l = splitted[1]
		txt = "".join(splitted[3:])
		txt = tokenize_line(txt)
		res.append((l, txt))
		if i == max_ex:
			break
	return res

def make_data_conv(all_line, max_ex):
	res = []

	for i, line in enumerate(all_line):
		if line == "":
			continue
		splitted = line.split(",")
		l = splitted[1]
		txt = "".join(splitted[3:])
		txt = tokenize_line_2(txt)
		res.append((l, txt))
		if i == max_ex:
			break
	return res
#splitted_data = make_data(all_line)
# print(splitted_data[0])
# print(splitted_data[6])
# print(splitted_data[7])
# print(splitted_data[15])
# print(splitted_data[60])
# print(splitted_data[75])
# print(splitted_data[153])
# print(splitted_data[179])
#print(splitted_data[32])
