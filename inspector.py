from flask import Flask, request, render_template
from werkzeug.routing import BaseConverter
import os
#os.environ['MPLCONFIGDIR'] = "/var/www/realec/.matplotlib"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import sys
import traceback
import re
from time import sleep,time

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import nltk
from collections import Counter, defaultdict
import itertools
import json
from operator import itemgetter
import string

import copy
from statistics import mean

nltk.data.path.append('/var/www/inspector/nltk_data/')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag, pos_tag_sents

from nltk import StanfordPOSTagger

import ufal.udpipe

import enchant
import enchant.checker
from enchant.checker import SpellChecker
from enchant.checker.CmdLineChecker import CmdLineChecker

class Model:
	def __init__(self, path):
		"""Load given model."""
		self.model = ufal.udpipe.Model.load(path)
		if not self.model:
			raise Exception("Cannot load UDPipe model from file '%s'" % path)

	def tokenize(self, text):
		"""Tokenize the text and return list of ufal.udpipe.Sentence-s."""
		tokenizer = self.model.newTokenizer(self.model.DEFAULT)
		if not tokenizer:
			raise Exception("The model does not have a tokenizer")
		return self._read(text, tokenizer)

	def read(self, text, in_format):
		"""Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
		input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
		if not input_format:
			raise Exception("Cannot create input format '%s'" % in_format)
		return self._read(text, input_format)

	def _read(self, text, input_format):
		input_format.setText(text)
		error = ufal.udpipe.ProcessingError()
		sentences = []

		sentence = ufal.udpipe.Sentence()
		while input_format.nextSentence(sentence, error):
			sentences.append(sentence)
			sentence = ufal.udpipe.Sentence()
		if error.occurred():
			raise Exception(error.message)

		return sentences

	def tag(self, sentence):
		"""Tag the given ufal.udpipe.Sentence (inplace)."""
		self.model.tag(sentence, self.model.DEFAULT)

	def parse(self, sentence):
		"""Parse the given ufal.udpipe.Sentence (inplace)."""
		self.model.parse(sentence, self.model.DEFAULT)

	def write(self, sentences, out_format):
		"""Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

		output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
		output = ''
		for sentence in sentences:
			output += output_format.writeSentence(sentence)
		output += output_format.finishDocument()

		return output

model = Model('english-partut-ud-2.0-170801.udpipe')

def check_spelling(text):
	chkr = enchant.checker.SpellChecker("en_GB")
	chkr.set_text(text)
	for err in chkr:
		#print (err.word)
		sug = err.suggest()[0]
		err.replace(sug)
	
	c = chkr.get_text()#returns corrected text
	return c

def space(string):
	string = re.sub('([a-zA-Z]| )([\.\?!])', '\\1\\2 ', string)
	string = re.sub(': ', ' : ', string)
	string = re.sub('; ', ' ; ', string)
	string = re.sub('  +', ' ', string)
	return string

def get_parsed_text(text):
	text = check_spelling(text)
	text = space(text)
	sentences = model.tokenize(text)
	for s in sentences:
		model.tag(s)
		model.parse(s)
	output = model.write(sentences, "conllu")
	return output

def count_tokens(parsed_text):
	num_tokens = 0
	lst_str = parsed_text.split('\n')
	for every_str in lst_str:
		#print(every_str)
		if ('PUNCT' not in every_str) and (every_str.startswith('#') == False) and (every_str != ''):
			#print(every_str)
			num_tokens += 1
			#print(num_tokens)
	return num_tokens

def count_sent(parsed_text):
	sent_lst = re.findall('(1\t.+?)\n\n', parsed_text, re.DOTALL)
	return len(sent_lst)

def parsing_things(string):
	token = re.search('[0-9]+\t(.+?)\t', string).group(1)
	order = re.search('([0-9]+)\t', string).group(1)
	head = re.search('\t([0-9]+)\t', string).group(1)
	rel_type = re.search('\t[0-9]+\t(.+?)\t', string).group(1)
	pos = re.search('[0-9]+\t.+?\t.+?\t(.+?)\t', string).group(1)
	#grammar = re.search('[VERB|AUX]\t.+?\t(.+?)\t', every_str).group(1)
	return order, token, head, rel_type, pos

def count_clauses_every_sent(parsed_text):
	sent_lst = re.findall('(1\t.+?)\n\n', parsed_text, re.DOTALL)
	verb_cl = {}
	all_num_sent = count_sent(parsed_text)
	for sent in range(1, all_num_sent+1):
		verb_cl[sent] = []
	num_sent = 1
	for every_sent in sent_lst:
		lst_str = every_sent.split('\n')
		for every_str in lst_str:
			if ('VerbForm=Fin' in every_str):
				sent_id = str(num_sent)
				order, token, head, rel_type, pos = parsing_things(every_str)
				if head not in verb_cl[int(sent_id)] and rel_type != 'conj':
					verb_cl[int(sent_id)].append([order, head])
		num_sent += 1
	for key, value in verb_cl.items():
		if verb_cl[key] == []:
			verb_cl[key] = [None]
		verb_cl[key] = len(verb_cl[key])
	return verb_cl

def count_clauses(parsed_text):
	verb_cl = count_clauses_every_sent(parsed_text)
	num_cl = 0
	#print(verb_cl)
	for key, value in verb_cl.items():
		num_cl = num_cl + value
	return num_cl

def count(d):
	num = 0
	for key, value in d.items():
		num = num + value
	return num

def count_np(parsed_text):
	sent_lst = re.findall('(1\t.+?)\n\n', parsed_text, re.DOTALL)
	all_num_sent = count_sent(parsed_text)
	adj_noun = {}
	poss = {}
	adj_vp = {}
	noun_vp = {}
	part_vp = {}
	for sent in range(1, all_num_sent+1):
		poss[sent] = 0
		adj_vp[sent] = []
		noun_vp[sent] = []
		part_vp[sent] = []
	num_sent = 1
	Colorize = []
	for every_sent in sent_lst:
		lst_str = every_sent.split('\n')
		c_start = len(Colorize) - 1
		adjo = []
		parto = []
		for every_str in lst_str:
			if 'nmod' in every_str:# включая nmod:poss
				order, token, head, rel_type, pos = parsing_things(every_str)
				poss[num_sent] = poss[num_sent] + 1
				Colorize.append((token,Green))
			elif 'ADP' in every_str:
				order, token, head, rel_type, pos = parsing_things(every_str)
				poss[num_sent] = poss[num_sent] + 1
				Colorize.append((token,Green))
			elif 'VerbForm=Ger' in every_str or 'VerbForm=Inf' in every_str and 'xcomp' not in every_str:
				order, token, head, rel_type, pos = parsing_things(every_str)
				poss[num_sent] = poss[num_sent] + 1
				Colorize.append((token,Green))
			elif 'ADJ' in every_str:
				order, token, head, rel_type, pos = parsing_things(every_str)
				adj_vp[num_sent].append(head)
				adjo.append(order)
				Colorize.append((token,None))
			elif 'VerbForm=Part' in every_str:
				order, token, head, rel_type, pos = parsing_things(every_str)
				part_vp[num_sent].append(head)
				parto.append(order)
				Colorize.append((token,None))
			elif 'NOUN' in every_str:
				order, token, head, rel_type, pos = parsing_things(every_str)
				noun_vp[num_sent].append(order)
				Colorize.append((token,None))
			else:
				order, token, head, rel_type, pos = parsing_things(every_str)
				Colorize.append((token,None))
		if num_sent in adj_vp and num_sent in noun_vp and len(adj_vp[num_sent]) > 0 and len(noun_vp[num_sent]) > 0:
			for i in range(min(len(adj_vp[num_sent]),len(noun_vp[num_sent]))):
				if adj_vp[num_sent][i] == noun_vp[num_sent][i]:
					poss[num_sent] = poss[num_sent] + 1
					Colorize[c_start+int(noun_vp[num_sent][i])] = (Colorize[c_start+int(noun_vp[num_sent][i])][0],Green)
					Colorize[c_start+int(adjo[i])] = (Colorize[c_start+int(adjo[i])][0],Green)
		if num_sent in part_vp and num_sent in noun_vp and len(part_vp[num_sent]) > 0 and len(noun_vp[num_sent]) > 0:
			for i in range(min(len(part_vp[num_sent]),len(noun_vp[num_sent]))):
				if part_vp[num_sent][i] == noun_vp[num_sent][i]:
					poss[num_sent] = poss[num_sent] + 1
					Colorize[c_start+int(noun_vp[num_sent][i])] = (Colorize[c_start+int(noun_vp[num_sent][i])][0],Green)
					Colorize[c_start+int(parto[i])] = (Colorize[c_start+int(parto[i])][0],Green)
		num_sent += 1
	return count(poss), Colorize

def find_ger_inf(parsed_text):
	Yellowish = '#ffd900'
	Reddish = '#ff3333'
	sent_lst = re.findall('(1\t.+?)\n\n', parsed_text, re.DOTALL)
	all_num_sent = count_sent(parsed_text)
	poss = {}
	Colorize = []
	for sent in range(1, all_num_sent+1):
		poss[sent] = 0
	num_sent = 1
	for every_sent in sent_lst:
		lst_str = every_sent.split('\n')
		for every_str in lst_str:
			if 'VerbForm=Ger' in every_str:
				order, token, head, rel_type, pos = parsing_things(every_str)
				poss[num_sent] = poss[num_sent] + 1
				Colorize.append((token,Reddish))
			elif 'VerbForm=Inf' in every_str and 'xcomp' not in every_str:
				order, token, head, rel_type, pos = parsing_things(every_str)
				poss[num_sent] = poss[num_sent] + 1
				Colorize.append((token,Yellowish))
			else:
				order, token, head, rel_type, pos = parsing_things(every_str)
				Colorize.append((token,None))
		num_sent += 1
	res = count(poss)
	return res, Colorize

Red = "#CC0000"
Green = "#00CC00"
Blue = "#0000CC"

def multisplit(text,array):
	output = []
	posarr = [0]
	posarr = posarr + array
	posarr = posarr + [len(text)]
	for i in range(1,len(posarr)):
		output.append(text[posarr[i-1]:posarr[i]])
	return output

def run_enchant(intext):
	s = SpellChecker('en_GB')
	s.set_text(intext)
	i = 0
	Colorize = []
	for error in s:
		i += 1
		Colorize.append((error.word, Red))
	return i, Colorize

wordnet_lemmatizer = WordNetLemmatizer()
st = StanfordPOSTagger('english-bidirectional-distsim.tagger','stanford-postagger.jar')
st.java_options = '-mx4096m'

folder = os.path.join(os.path.dirname(__file__), 'lists')
ielts_freq_path = os.path.join(folder, 'ielts_freq.txt')
ielts_fixed_spelling_path = os.path.join(folder, 'ielts_fixed_spelling.txt')
english_voc_min_level_path = os.path.join(folder, 'english_voc_min_level.txt')
wff_min_level_path = os.path.join(folder, 'wff_min_level_no_x.txt')
coca3000dict_path = os.path.join(folder, 'coca3000dict.txt')
academic_wf_path = os.path.join(folder, 'academic_wf.txt')
awl_all_path = os.path.join(folder, 'awl_all.txt')
stopwords_eng_path = os.path.join(folder, 'stopwords_eng.txt')
mean_sent_lengths_path = os.path.join(folder, 'mean_sen_lengths.txt')
mean_word_lengths_path = os.path.join(folder, 'mean_word_lengths.txt')
introwords_path = os.path.join(folder, 'introwords')
introwords_hash_path = os.path.join(folder, 'intro_words_hash')
pearson_hash_path = os.path.join(folder, 'pearson_hash')

translator = str.maketrans(dict.fromkeys(string.punctuation))

with open(ielts_freq_path, 'r', encoding='utf-8') as f:
	ielts = Counter(json.load(f))

with open(ielts_fixed_spelling_path, 'r', encoding='utf-8') as f:
	ielts_fixed = Counter(json.load(f))
	
with open(english_voc_min_level_path, 'r', encoding='utf-8') as f:
	eng_voc = Counter(json.load(f))
	
with open(wff_min_level_path, 'r', encoding='utf-8') as f:
	wff = Counter(json.load(f))
	
with open(coca3000dict_path, 'r', encoding='utf-8') as f:
	COCA_freq = Counter(json.load(f))

with open(academic_wf_path, 'r', encoding='utf-8') as f:
	COCA_academic = set(f.read().split())

awl = set()
with open(awl_all_path, 'r', encoding='utf-8') as f:
	for line in f.readlines():
		awl.add(line.strip())

stopwords_eng = set()
with open(stopwords_eng_path, 'r', encoding='utf-8') as f:
	for line in f.readlines():
		stopwords_eng.add(line.strip())

mean_sent_lengths = np.loadtxt(mean_sent_lengths_path)
mean_word_lengths = np.loadtxt(mean_word_lengths_path)

introwords = set()
with open(introwords_path, 'r', encoding='utf-8') as f:
	for line in f.readlines():
		introwords.add(line.strip())

with open(introwords_hash_path, 'r', encoding='utf-8') as data_file:	
	introwords_hash = json.load(data_file)

with open(pearson_hash_path, 'r', encoding='utf-8') as f:
	pearson_hash_table = Counter(json.load(f))

def isNum(token):
	try:
		int(token)
		return True
	except:
		return False

def GetNearest(val, valarr):
	Nearest = valarr[0]
	Diff = abs(val-Nearest)
	for value in sorted(valarr):
		if abs(val-value) < Diff:
			Diff = abs(val-value)
			Nearest = value
	return Nearest

from nltk.corpus import wordnet as wn

def CountUniques(tagged):
	s = SpellChecker('en_GB')
	uniques = []
	Colorize = []
	tokensarr = tagged
	result = 0
	for sentence in tokensarr:
		for token in sentence:
			if len(token) < 2:
				continue
			word = token[0].lower()
			if not re.search(r'\w',word):
				continue
			if not s.check(word):
				try:
					word = s.suggest(word)[0]
				except:
					pass
			p = wn.NOUN
			if token[1][0] == 'J':
				p = wn.ADJ
			elif token[1][0] == 'R':
				p = wn.ADV
			elif token[1][0] == 'V':
				p = wn.VERB
			word = wordnet_lemmatizer.lemmatize(word, pos = p)
			if word not in uniques:
				uniques.append(word)
				result += 1
				Colorize.append((token[0],Red))
			else:
				Colorize.append((token[0],None))
	return result, Colorize

def ReadDataJson(filename):
	PROJECT_ROOT = os.path.realpath(os.path.dirname(__file__))
	json_url = os.path.join(PROJECT_ROOT, filename)
	with open(json_url,'r',encoding='utf-8') as j:
		data = json.load(j)
	return data

def GenerateEvaluation(ValuesDict):
	Marks = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
	predicted_mark = GetNearest(ValuesDict['predicted_mark'],Marks)
	if ValuesDict['type'] == 'graph':
		etalons = ReadDataJson('graph.json')
		
		if etalons[str(predicted_mark)]["MarkOverflow"]:
			ResultPage = """<tr id="wow" class="green"><td style="width: 100%"><b>Wow! It seems that we do not even have essays rated as high as yours, so we will have to compare it to the nearest-rated.<b></td></tr>"""
		else:
			ResultPage = ""

		### NUM_TOKENS
		ResultPage += """			<tr id="tokens_tr" """

		if ValuesDict['num_tokens']<etalons[str(predicted_mark)]['num_tokens'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Total words
								</td>
								<td style="width: 100%">
										Even though <a class="innerlink" id="tokenslink" onclick="popupbox(event,'tokens')" href="javascript:void(0)">your essay length</a> is bigger than required, it still measured too short by our means. Consider composing longer texts, as they tend to be rated higher
								</td>
						</tr>"""
		elif ValuesDict['num_tokens']<etalons[str(predicted_mark)]['num_tokens'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Total words
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="tokenslink" onclick="popupbox(event,'tokens')" href="javascript:void(0)">The amount of words in your text</a> is average. However, remember that longer texts tend to be rated higher
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Total words
								</td>
								<td style="width: 100%">
										Your text has <a class="innerlink" id="tokenslink" onclick="popupbox(event,'tokens')" href="javascript:void(0)">a fair number of words</a>, which is great, since longer texts tend to be rated higher
								</td>
						</tr>"""
		
		
		### NUM_0
		ResultPage += """			<tr id="num_0_tr" """
		
		if ValuesDict['num_0']<etalons[str(predicted_mark)]['num_0'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Unrecognized words
								</td>
								<td style="width: 100%">
										Your text shows a fairly low quantity of <a class="innerlink" id="num_0link" onclick="reversemaskerlink(this,'num_0')" href="javascript:showmasker('num_0')">words beyond our dictionary volume</a>. That is a good feature: texts with big amounts of such words (which could be erroneous, non-existent or too rare) tend to be rated lower
								</td>
						</tr>"""
		elif ValuesDict['num_0']<etalons[str(predicted_mark)]['num_0'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Unrecognized words
								</td>
								<td style="width: 100%">
										There is a medium amount of <a class="innerlink" id="num_0link" onclick="reversemaskerlink(this,'num_0')" href="javascript:showmasker('num_0')">words we could not match to our dictionary data</a>. Keep in mind that these words negatively affect your grade: they may be erroneous, non-existent or too rare
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Unrecognized words
								</td>
								<td style="width: 100%">
										We detected a big number of <a class="innerlink" id="num_0link" onclick="reversemaskerlink(this,'num_0')" href="javascript:showmasker('num_0')">words we could not match to our dictionary data</a>. It may negatively affect your grade: these words may be erroneous, non-existent or too rare. Consider sticking to more standard words and expressions
								</td>
						</tr>"""


		### MEANWORDLENGTH
		ResultPage += """			<tr id="MeanWordLength_tr" """

		if ValuesDict['MeanWordLength']<etalons[str(predicted_mark)]['MeanWordLength'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Word length
								</td>
								<td style="width: 100%">
										Your <a class="innerlink" id="MeanWordLengthlink" onclick="popupbox(event,'MeanWordLength')" href="javascript:void(0)">mean word length</a> is too short. Consider using more complex words: this contributes to the good grade.
								</td>
						</tr>"""
		elif ValuesDict['MeanWordLength']<etalons[str(predicted_mark)]['MeanWordLength'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Word length
								</td>
								<td style="width: 100%">
										Your <a class="innerlink" id="MeanWordLengthlink" onclick="popupbox(event,'MeanWordLength')" href="javascript:void(0)">mean word length</a> is average. The bigger it gets, though, the higher your grade may be.
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Word length
								</td>
								<td style="width: 100%">
										Your words are in general <a class="innerlink" id="MeanWordLengthlink" onclick="popupbox(event,'MeanWordLength')" href="javascript:void(0)">sufficiently long</a>. Keep it up, as mean word length has shown to be positively correlated to the grade
								</td>
						</tr>"""
		
		
		### NUM_SENTS
		ResultPage += """			<tr id="sents_tr" """
		
		if ValuesDict['num_sents']<etalons[str(predicted_mark)]['num_sents'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Sentences ratio
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="sentslink" onclick="popupbox(event,'sents')" href="javascript:void(0)">As compared</a> to total number of words in your text, the number of sentences in your text is fairly low. That is great, given that your sentences are well-structured and not too short. This contributes well to the mark you may hope to get
								</td>
						</tr>"""
		elif ValuesDict['num_sents']<etalons[str(predicted_mark)]['num_sents'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Sentences ratio
								</td>
								<td style="width: 100%">
										 <a class="innerlink" id="sentslink" onclick="popupbox(event,'sents')" href="javascript:void(0)">Your number of sentences divided by number of words</a> is average for text of your type. Keep in mind that the lower the number of sentences, the better grade can be expected
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Sentences ratio
								</td>
								<td style="width: 100%">
										It seems that you have used <a class="innerlink" id="sentslink" onclick="popupbox(event,'sents')" href="javascript:void(0)">a lot</a> of sentences compared to the number of words. Our data has shown that this may in fact negatively affect your grade, so consider restructuring your text
								</td>
						</tr>"""
		
		
		### NUM_STOPWORDS
		ResultPage += """			<tr id="stopwords_tr" """
		
		if ValuesDict['num_stopwords']<etalons[str(predicted_mark)]['num_stopwords'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Too common words
								</td>
								<td style="width: 100%">
										You did a great job of limiting the use of <a class="innerlink" id="stopwordslink" onclick="reversemaskerlink(this,'stopwords')" href="javascript:showmasker('stopwords')">the most commonly used words</a>. Maintain your vocabulary diversity: this will improve your grade
								</td>
						</tr>"""
		elif ValuesDict['num_stopwords']<etalons[str(predicted_mark)]['num_stopwords'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Too common words
								</td>
								<td style="width: 100%">
										The number of <a class="innerlink" id="stopwordslink" onclick="reversemaskerlink(this,'stopwords')" href="javascript:showmasker('stopwords')">most commonly used words</a> in your text is within average. Note that using more complex vocabulary instead of more basic affects your mark positively
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Too common words
								</td>
								<td style="width: 100%">
										The value of <a class="innerlink" id="stopwordslink" onclick="reversemaskerlink(this,'stopwords')" href="javascript:showmasker('stopwords')">the most commonly used words</a> in the text is pretty high. Consider using more diverse vocabulary as a way to secure a good grade
								</td>
						</tr>"""


		### NUM_INTRO_GROUPS
		ResultPage += """			<tr id="intro_groups_tr" """

		if ValuesDict['num_intro_groups']<etalons[str(predicted_mark)]['num_intro_groups'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Inroductory units
								</td>
								<td style="width: 100%">
										We detected a pretty low number of <a class="innerlink" id="intro_groupslink" onclick="reversemaskerlink(this,'intro_groups')" href="javascript:showmasker('intro_groups')">inroductory units</a> in your text. That is not good, as they contribute to the grade. Consider using more of them
								</td>
						</tr>"""
		elif ValuesDict['num_intro_groups']<etalons[str(predicted_mark)]['num_intro_groups'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Inroductory units
								</td>
								<td style="width: 100%">
										We detected an average number of <a class="innerlink" id="intro_groupslink" onclick="reversemaskerlink(this,'intro_groups')" href="javascript:showmasker('intro_groups')">inroductory units</a> used in your text. They contribute to the good grade, so keep on using them
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Inroductory units
								</td>
								<td style="width: 100%">
										We detected a high number of <a class="innerlink" id="intro_groupslink" onclick="reversemaskerlink(this,'intro_groups')" href="javascript:showmasker('intro_groups')">inroductory units</a> in your text. That is good, keep it up!
								</td>
						</tr>"""
		
		
		### MAXSENTLENGTH
		ResultPage += """			<tr id="MaxSentLength_tr" """
		
		if ValuesDict['MaxSentLength']<etalons[str(predicted_mark)]['MaxSentLength'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Longest sentence
								</td>
								<td style="width: 100%">
										You have done well avoiding using too long sentences in your text, judging by your <a class="innerlink" id="MaxSentLengthlink" onclick="popupbox(event,'MaxSentLength')" href="javascript:void(0)">maximum sentence length</a>. Continue  using well-structured though not very long sentences
								</td>
						</tr>"""
		elif ValuesDict['MaxSentLength']<etalons[str(predicted_mark)]['MaxSentLength'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Longest sentence
								</td>
								<td style="width: 100%">
										Your longest sentence is <a class="innerlink" id="MaxSentLengthlink" onclick="popupbox(event,'MaxSentLength')" href="javascript:void(0)">of an average length</a>. We suggest trying not to make your text overcomplicated, as very long sentences have shown a decreasing effect on the grade
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Longest sentence
								</td>
								<td style="width: 100%">
										Your longest sentence is <a class="innerlink" id="MaxSentLengthlink" onclick="popupbox(event,'MaxSentLength')" href="javascript:void(0)">too long</a> by our standards. Avoid using too long sentences as they affect the understanding of the text and — as a result — the grade
								</td>
						</tr>"""
		
		
		### NUM_NP
		ResultPage += """			<tr id="np_tr" """
		
		if ValuesDict['num_np']<etalons[str(predicted_mark)]['num_np'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Noun phrases
								</td>
								<td style="width: 100%">
										The <a class="innerlink" id="nplink" onclick="reversemaskerlink(this,'np')" href="javascript:showmasker('np')">overuse</a> of noun phrases has shown negative effect, according to our data. Consider using different phrases in expressing your ideas
								</td>
						</tr>"""
		elif ValuesDict['num_np']<etalons[str(predicted_mark)]['num_np'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Noun phrases
								</td>
								<td style="width: 100%">
										You have used an average number of <a class="innerlink" id="nplink" onclick="reversemaskerlink(this,'np')" href="javascript:showmasker('np')">noun phrases</a> in your text. Try not to overuse them, though: our data have shown it to be a bad feature
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Noun phrases
								</td>
								<td style="width: 100%">
										We have not found too much <a class="innerlink" id="nplink" onclick="reversemaskerlink(this,'np')" href="javascript:showmasker('np')">noun phrases</a> in your text, which turns out to be good: they have been proved to affect the grade
								</td>
						</tr>"""
		
		
		### NUM_CL
		ResultPage += """			<tr id="cl_tr" """
		
		if ValuesDict['num_cl']<etalons[str(predicted_mark)]['num_cl'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Simple sentences
								</td>
								<td style="width: 100%">
										Your text consists of pretty <a class="innerlink" id="cllink" onclick="popupbox(event,'cl')" href="javascript:void(0)">too many</a> simple sentences, or clauses. Try not to compose too long texts: express your ideas in a compact way
								</td>
						</tr>"""
		elif ValuesDict['num_cl']<etalons[str(predicted_mark)]['num_cl'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Simple sentences
								</td>
								<td style="width: 100%">
										Your text consists of <a class="innerlink" id="cllink" onclick="popupbox(event,'cl')" href="javascript:void(0)">a moderate amount</a> of simple sentences, or clauses. Stick to expressing your ideas in a compact way and avoid composing too long texts
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Simple sentences
								</td>
								<td style="width: 100%">
										You have <a class="innerlink" id="cllink" onclick="popupbox(event,'cl')" href="javascript:void(0)">not too many</a> simple sentences (clauses) in your text. This is fine: too lengthy texts score less, as our data shows. Just make sure to outline all the necessary points in your text.
								</td>
						</tr>"""


		### VERBS_QUAN
		
		ResultPage += """			<tr id="verbs_quan_tr" """
		if ValuesDict['verbs_quan']<etalons[str(predicted_mark)]['verbs_quan'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Verb usage
								</td>
								<td style="width: 100%">
										Highly rated essays have a bigger proportion of <a class="innerlink" id="verbslink" onclick="reversemaskerlink(this,'verbs')" href="javascript:showmasker('verbs')"><div class="yellowish" style="display: inline-block">infinitival</div>, <div class="reddish" style="display: inline-block">gerundival</div> and <div class="blueish" style="display: inline-block">participial</div></a> constructions than yours. Remember the need to apply them when you write your next text
								</td>
						</tr>"""
		elif ValuesDict['verbs_quan']<etalons[str(predicted_mark)]['verbs_quan'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Verb usage
								</td>
								<td style="width: 100%">
										You have used <a class="innerlink" id="verbslink" onclick="reversemaskerlink(this,'verbs')" href="javascript:showmasker('verbs')"><div class="yellowish" style="display: inline-block">infinitival</div>, <div class="reddish" style="display: inline-block">gerundival</div> and <div class="blueish" style="display: inline-block">participial</div></a> constructions in moderation. Note that our research has shown that their quantity positively correlates with good grades
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Verb usage
								</td>
								<td style="width: 100%">
										You made a good use of <a class="innerlink" id="verbslink" onclick="reversemaskerlink(this,'verbs')" href="javascript:showmasker('verbs')"><div class="yellowish" style="display: inline-block">infinitival</div>, <div class="reddish" style="display: inline-block">gerundival</div> and <div class="blueish" style="display: inline-block">participial</div></a> constructions in your text. Well done, this should improve your grade!
								</td>
						</tr>"""


	else:
		etalons = ReadDataJson('opinion.json')
		
		if etalons[str(predicted_mark)]["MarkOverflow"]:
			ResultPage = """<tr id="wow" class="green"><td style="width: 100%"><b>Wow! It seems that we do not even have essays rated as high as yours, so we will have to compare it to the nearest-rated.<b></td></tr>"""
		else:
			ResultPage = ""

		### NUM_TOKENS
		ResultPage += """			<tr id="tokens_tr" """

		if ValuesDict['num_tokens']<etalons[str(predicted_mark)]['num_tokens'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Total words
								</td>
								<td style="width: 100%">
										Even though <a class="innerlink" id="tokenslink" onclick="popupbox(event,'tokens')" href="javascript:void(0)">your essay length</a> is bigger than required, it still measured too short by our means. Consider composing longer texts, as they tend to be rated higher.
								</td>
						</tr>"""
		elif ValuesDict['num_tokens']<etalons[str(predicted_mark)]['num_tokens'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Total words
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="tokenslink" onclick="popupbox(event,'tokens')" href="javascript:void(0)">The amount of words in your text</a> is moderate, which is not bad. However, remember that longer texts tend to be rated higher
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Total words
								</td>
								<td style="width: 100%">
										Your text has <a class="innerlink" id="tokenslink" onclick="popupbox(event,'tokens')" href="javascript:void(0)">a fair number of words</a>, which is great, since longer texts tend to be rated higher
								</td>
						</tr>"""
		
		
		### SPELL_Q
		ResultPage += """			<tr id="spell_q_tr" """
		
		if ValuesDict['spell_q']<etalons[str(predicted_mark)]['spell_q'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Misspelled words
								</td>
								<td style="width: 100%">
										We have detected a fairly low quantity of <a class="innerlink" id="spell_qlink" onclick="reversemaskerlink(this,'spell_q')" href="javascript:showmasker('spell_q')">misspelled words</a> in your text. That is good, keep it up!
								</td>
						</tr>"""
		elif ValuesDict['spell_q']<etalons[str(predicted_mark)]['spell_q'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Misspelled words
								</td>
								<td style="width: 100%">
										There is a moderate amount of <a class="innerlink" id="spell_qlink" onclick="reversemaskerlink(this,'spell_q')" href="javascript:showmasker('spell_q')">misspelled words</a> we have detected in the text. Keep in mind to check your texts for these.
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Misspelled words
								</td>
								<td style="width: 100%">
										We detected a high amount of <a class="innerlink" id="spell_qlink" onclick="reversemaskerlink(this,'spell_q')" href="javascript:showmasker('spell_q')">misspelled words</a> in your text. That negatively affects the mark: remember to thoroughly check your texts
								</td>
						</tr>"""


		### MEANWORDLENGTH
		ResultPage += """			<tr id="MeanWordLength_tr" """

		if ValuesDict['MeanWordLength']<etalons[str(predicted_mark)]['MeanWordLength'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Word length
								</td>
								<td style="width: 100%">
										Your <a class="innerlink" id="MeanWordLengthlink" onclick="popupbox(event,'MeanWordLength')" href="javascript:void(0)">mean word length</a> is too short. Consider using more complex words: this contributes to the mark
								</td>
						</tr>"""
		elif ValuesDict['MeanWordLength']<etalons[str(predicted_mark)]['MeanWordLength'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Word length
								</td>
								<td style="width: 100%">
										Your <a class="innerlink" id="MeanWordLengthlink" onclick="popupbox(event,'MeanWordLength')" href="javascript:void(0)">mean word length</a> is pretty average. The bigger it gets, though, the higher your mark is expected to be
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Total words
								</td>
								<td style="width: 100%">
										Your words are in general <a class="innerlink" id="MeanWordLengthlink" onclick="popupbox(event,'MeanWordLength')" href="javascript:void(0)">sufficiently long</a>. Keep it up, as mean word length has shown to positively affect the mark
								</td>
						</tr>"""


		### VERBS_QUAN
		
		ResultPage += """			<tr id="verbs_quan_tr" """
		if ValuesDict['verbs_quan']<etalons[str(predicted_mark)]['verbs_quan'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Verb usage
								</td>
								<td style="width: 100%">
										Highly rated essays have a bigger proportion of <a class="innerlink" id="verbslink" onclick="reversemaskerlink(this,'verbs')" href="javascript:showmasker('verbs')"><div class="yellowish" style="display: inline-block">infinitival</div>, <div class="reddish" style="display: inline-block">gerundival</div> and <div class="blueish" style="display: inline-block">participial</div></a> constructions. Keep the need to apply them in mind when you write your next essay
								</td>
						</tr>"""
		elif ValuesDict['verbs_quan']<etalons[str(predicted_mark)]['verbs_quan'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Verb usage
								</td>
								<td style="width: 100%">
										You have used <a class="innerlink" id="verbslink" onclick="reversemaskerlink(this,'verbs')" href="javascript:showmasker('verbs')"><div class="yellowish" style="display: inline-block">infinitival</div>, <div class="reddish" style="display: inline-block">gerundival</div> and <div class="blueish" style="display: inline-block">participial</div></a> moderately. Note that our research has shown that their quantity positively correlates with the mark
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Verb usage
								</td>
								<td style="width: 100%">
										You made a good use of <a class="innerlink" id="verbslink" onclick="reversemaskerlink(this,'verbs')" href="javascript:showmasker('verbs')"><div class="yellowish" style="display: inline-block">infinitival</div>, <div class="reddish" style="display: inline-block">gerundival</div> and <div class="blueish" style="display: inline-block">participial</div></a> constructions in your text. Well done, this should increase your mark!
								</td>
						</tr>"""
		
		
		### NUM_UNIQUE
		ResultPage += """			<tr id="unique_tr" """
		
		if ValuesDict['num_unique']<etalons[str(predicted_mark)]['num_unique'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Unique words
								</td>
								<td style="width: 100%">
										The quantity of <a class="innerlink" id="uniquelink" onclick="reversemaskerlink(this,'unique')" href="javascript:showmasker('unique')">unique words</a> in your text is rather low. This is surprisingly good: we suggest you elaborate on the topic in a suitable way. Keep in mind this correlation
								</td>
						</tr>"""
		elif ValuesDict['num_unique']<etalons[str(predicted_mark)]['num_unique'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Unique words
								</td>
								<td style="width: 100%">
										The quantity of <a class="innerlink" id="uniquelink" onclick="reversemaskerlink(this,'unique')" href="javascript:showmasker('unique')">unique words</a> in your text is moderate. Higher quantity of such words has surprisingly shown to lower the mark: we suggest you elaborate on the topic in a suitable way. Keep in mind this correlation
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Unique words
								</td>
								<td style="width: 100%">
										The quantity of <a class="innerlink" id="uniquelink" onclick="reversemaskerlink(this,'unique')" href="javascript:showmasker('unique')">unique words</a> in your text is high. According to the data of our research, this turns out to be bad for your text; we suggest you to find the possibilities for repeating some meaningful words, elaborating on the topic in a suitable way
								</td>
						</tr>"""
		
		
		### NUM_MOST_COMMON_REPETITION
		ResultPage += """			<tr id="most_common_repetition_tr" """
		
		if ValuesDict['num_most_common_repetition']<etalons[str(predicted_mark)]['num_most_common_repetition'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Most common repetition
								</td>
								<td style="width: 100%">
										 Your <a class="innerlink" id="most_common_repetitionlink" onclick="reversemaskerlink(this,'most_common_repetition')" href="javascript:showmasker('most_common_repetition')">most repeated word</a> appears in your text pretty rarely. That is great, as too common repetitions negatvely affect the mark.
								</td>
						</tr>"""
		elif ValuesDict['num_most_common_repetition']<etalons[str(predicted_mark)]['num_most_common_repetition'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Most common repetition
								</td>
								<td style="width: 100%">
										The quantity of your <a class="innerlink" id="most_common_repetitionlink" onclick="reversemaskerlink(this,'most_common_repetition')" href="javascript:showmasker('most_common_repetition')">most commonly repeated word</a> is pretty average. Avoid repeating certain words too much: this negatvely affects the mark
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Most common repetition
								</td>
								<td style="width: 100%">
										Your <a class="innerlink" id="most_common_repetitionlink" onclick="reversemaskerlink(this,'most_common_repetition')" href="javascript:showmasker('most_common_repetition')">most repeated word</a> appears in your text too often. Consider avoiding the overuse of certain words
								</td>
						</tr>"""
		
		
		### NUM_B1
		ResultPage += """			<tr id="b1_tr" """
		
		if ValuesDict['num_b1']<etalons[str(predicted_mark)]['num_b1'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Words of B1 level
								</td>
								<td style="width: 100%">
										You <a class="innerlink" id="b1link" onclick="reversemaskerlink(this,'b1')" href="javascript:showmasker('b1')">have not used</a> too many words of advanced level. Try to include more of them, as it has proved positive on the mark
								</td>
						</tr>"""
		elif ValuesDict['num_b1']<etalons[str(predicted_mark)]['num_b1'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Words of B1 level
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="b1link" onclick="reversemaskerlink(this,'b1')" href="javascript:showmasker('b1')">Your usage</a> of words of advanced level is well within our averages. Using more of these gives a significant boon to your mark, as shown by our data
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Words of B1 level
								</td>
								<td style="width: 100%">
										You have used <a class="innerlink" id="b1link" onclick="reversemaskerlink(this,'b1')" href="javascript:showmasker('b1')">quite a lot</a> of words of advanced level. It is great: the more you use them, the higher mark you get
								</td>
						</tr>"""
		
		
		### TIMEANDSEQUENCE
		ResultPage += """			<tr id="TimeAndSequence_tr" """
		
		if ValuesDict['TimeAndSequence']<etalons[str(predicted_mark)]['TimeAndSequence'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Constructions of time
								</td>
								<td style="width: 100%">
										You have not expressed time notions in your text <a class="innerlink" id="TimeAndSequencelink" onclick="reversemaskerlink(this,'TimeAndSequence')" href="javascript:showmasker('TimeAndSequence')">too often</a>. This is actually beneficial for your mark, as shown by our research. You have done well by not overusing them
								</td>
						</tr>"""
		elif ValuesDict['TimeAndSequence']<etalons[str(predicted_mark)]['TimeAndSequence'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Constructions of time
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="TimeAndSequencelink" onclick="reversemaskerlink(this,'TimeAndSequence')" href="javascript:showmasker('TimeAndSequence')">Your usage</a> of time and sequence construction is pretty normal. According to our data, this is a hidden feature which might lower your mark, so do not overuse them
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Constructions of time
								</td>
								<td style="width: 100%">
										You have used a lot of <a class="innerlink" id="TimeAndSequencelink" onclick="reversemaskerlink(this,'TimeAndSequence')" href="javascript:showmasker('TimeAndSequence')">time and sequence constructions</a>. That is not very good, as our research has shown that their usage is an underlying feature which affects the mark negatively
								</td>
						</tr>"""


		### MAXSENTLENGTH
		ResultPage += """			<tr id="MaxSentLength_tr" """
		
		if ValuesDict['MaxSentLength']<etalons[str(predicted_mark)]['MaxSentLength'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Longest sentence
								</td>
								<td style="width: 100%">
										You have done <a class="innerlink" id="MaxSentLengthlink" onclick="popupbox(event,'MaxSentLength')" href="javascript:void(0)">well</a> avoiding using too long sentences in your text. Continue using well-structured though not very long sentences
								</td>
						</tr>"""
		elif ValuesDict['MaxSentLength']<etalons[str(predicted_mark)]['MaxSentLength'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Longest sentence
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="MaxSentLengthlink" onclick="popupbox(event,'MaxSentLength')" href="javascript:void(0)">Your longest sentence</a> is pretty average. We suggest not to overcomplicate the text, as very long sentences have shown decreasing the mark 
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Longest sentence
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="MaxSentLengthlink" onclick="popupbox(event,'MaxSentLength')" href="javascript:void(0)">Your longest sentence</a> is too complex by our means. Avoid using too long sentences as overcomplicated structures negatively affect the understanding of the text as well as the mark
								</td>
						</tr>"""
		
		
		### NUM_SENTS
		ResultPage += """			<tr id="sents_tr" """
		
		if ValuesDict['num_sents']<etalons[str(predicted_mark)]['num_sents'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Sentences ratio
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="sentslink" onclick="popupbox(event,'sents')" href="javascript:void(0)">As compared</a> to total number of words in your text, you have not written too many sentences. That is a good strategy, keep it up, as elaborate though not too long sentences are the key to a better mark
								</td>
						</tr>"""
		elif ValuesDict['num_sents']<etalons[str(predicted_mark)]['num_sents'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Sentences ratio
								</td>
								<td style="width: 100%">
										Your number of sentences divided by number of words shows a <a class="innerlink" id="sentslink" onclick="popupbox(event,'sents')" href="javascript:void(0)">figure</a> which lays within average numbers for texts of your level. More elaborate (though not very long) sentences are the key to increasing your mark
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Sentences ratio
								</td>
								<td style="width: 100%">
										The number of sentences <a class="innerlink" id="sentslink" onclick="popupbox(event,'sents')" href="javascript:void(0)">you have</a> in your text compared to your number of words is high. This is a negative factor for your mark. Consider using more elaborate, though not very long sentences
								</td>
						</tr>"""


		### NUM_C1
		ResultPage += """			<tr id="c1_tr" """
		
		if ValuesDict['num_c1']<etalons[str(predicted_mark)]['num_c1'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Words of C1 level
								</td>
								<td style="width: 100%">
										The <a class="innerlink" id="c1link" onclick="reversemaskerlink(this,'c1')" href="javascript:showmasker('c1')">number</a> of words of C1 level in your text is low. We have found that sophisticated lexics contributes to your result, so you may try to use more of these
								</td>
						</tr>"""
		elif ValuesDict['num_c1']<etalons[str(predicted_mark)]['num_c1'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Words of C1 level
								</td>
								<td style="width: 100%">
										The words of C1 level in your text <a class="innerlink" id="c1link" onclick="reversemaskerlink(this,'c1')" href="javascript:showmasker('c1')">form</a> a pretty standard part of your text by our means. Note that using more of sophisticated lexics significantly improves your result
								</td>
						</tr>"""
		else:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Words of C1 level
								</td>
								<td style="width: 100%">
										We detected <a class="innerlink" id="c1link" onclick="reversemaskerlink(this,'c1')" href="javascript:showmasker('c1')">an impressive number</a> of words of C1 level in your text. Sophisticated lexics has shown exceptionally beneficial for the result on our data, so try to maintain these numbers
								</td>
						</tr>"""


		### MAXWORDLENGTH
		ResultPage += """			<tr id="MaxWordLength_tr" """
		
		if ValuesDict['MaxWordLength']<etalons[str(predicted_mark)]['MaxWordLength'][0]:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Longest word
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="MaxWordLengthlink" onclick="popupbox(event,'MaxWordLength')" href="javascript:void(0)">The largest word</a> of your text is rather too short for texts of your level. This may point to the lack of complex lexics in your text, so try to use more lengthy words
								</td>
						</tr>"""
		elif ValuesDict['MaxWordLength']<etalons[str(predicted_mark)]['MaxWordLength'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Longest word
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="MaxWordLengthlink" onclick="popupbox(event,'MaxWordLength')" href="javascript:void(0)">The longest word</a> of your text is nor too short nor too long. This simple metric has shown to positively correlate with the mark, so do not hesitate from using complex lexics
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Longest word
								</td>
								<td style="width: 100%">
										<a class="innerlink" id="MaxWordLengthlink" onclick="popupbox(event,'MaxWordLength')" href="javascript:void(0)">Your longest word</a> has measured an impressive result on our data. This metric allows to assess the complexity of your vocabulary, which, in part, positively correlates with the mark
								</td>
						</tr>"""


		### GER_INF
		ResultPage += """			<tr id="ger_inf_tr" """
		
		if ValuesDict['ger_inf']<etalons[str(predicted_mark)]['ger_inf'][0]:
				ResultPage += """class="green">
								<td valign="top" style="width: 12%">
										Gerunds and infinitives
								</td>
								<td style="width: 100%">
										The usage of gerunds and infinitives positively affects the mark, and you have done <a class="innerlink" id="c1link" onclick="reversemaskerlink(this,'ger_inf')" href="javascript:showmasker('ger_inf')">well</a> on this account
								</td>
						</tr>"""
		elif ValuesDict['ger_inf']<etalons[str(predicted_mark)]['ger_inf'][1]:
				ResultPage += """class="yellow">
								<td valign="top" style="width: 12%">
										Gerunds and infinitives
								</td>
								<td style="width: 100%">
										The usage of gerunds and infinitives positively affects the mark, and your text has measured a <a class="innerlink" id="c1link" onclick="reversemaskerlink(this,'ger_inf')" href="javascript:showmasker('ger_inf')">moderate number</a> of them
								</td>
						</tr>"""
		else:
				ResultPage += """class="red">
								<td valign="top" style="width: 12%">
										Gerunds and infinitives
								</td>
								<td style="width: 100%">
										We suggest you to use more gerunds and infinitives, as their usage positively affects the mark, and your text has measured a <a class="innerlink" id="c1link" onclick="reversemaskerlink(this,'ger_inf')" href="javascript:showmasker('ger_inf')">low number</a> of them
								</td>
						</tr>"""
		
	return ResultPage

class Token(object):
	def __init__(self, word):
		self.word = word
		self.iscefr = False
		self.isAcademic = False
		self.isNum = False
		self.isStopword = False
		self.isFreq = False
		self.cefr = None
		self.CEFR_bg_color = {'A1' : '#ffff99', 'A2' : '#ffcc33',
							'B1' : '#a6ff4d', 'B2' : '#668cff',
							'C1' : '#ff6666', 'C2' : '#ff00bf', 0 : '#000000'}
		self.FREQ_bg_color = {'1-500' : '#ffb3e6', '501-3000' : '#b3ffb3',
							0 : '#b3b3ff'}

	def setTokenCEFR(self, level):
		self.cefr = level
		self.iscefr = True

	def setTokenNum(self):
		self.isNum = True

	def setTokenStopword(self):
		self.isStopword = True

	def setTokenAcademic(self, bool_val):
		self.isAcademic = bool_val

	def setTokenFreq(self, freq):
		self.freq = freq
		self.isFreq = True

	def isTokenCEFR(self):
		return self.iscefr

	def isTokenAcademic(self):
		return self.isAcademic

	def getWord(self):
		return self.word

	def getLevel(self):
		return self.cefr

	def getColor(self):
		return self.CEFR_bg_color[self.cefr]

	def __str__(self):
		return self.word

	def printHTML(self, color, background_color):
		return '<span style="background-color: ' + str(background_color) + '">'+ self.word +'.</span>'

	def printHtmlFreq(self):
		background_color = self.FREQ_bg_color[self.freq]
		return '<span style="background-color: ' + str(background_color) + '">'+ self.word +'</span>'

	def print_word_cefr(self, level):
		if not self.iscefr:
			return self.word
		elif self.cefr == level:
			return '<font color="#002080">'+ self.word +'</font>'
		else:
			return self.word

	def is_cefr(self, level):
		if not self.iscefr:
			return 0
		elif self.cefr == level:
			return 1
		else:
			return 0

	def print_word_freq(self, freq):
		if not self.isFreq:
			return self.word
		elif self.freq == freq:
			return '<font color="#002080">'+ self.word +'</font>'
		else:
			return self.word

	def is_freq(self, freq):
		if not self.isFreq:
			return 0
		elif self.freq == freq:
			return 1
		else:
			return 0

	def print_word_academic(self):
		if self.isAcademic:
			return '<font color="#002080">'+ self.word +'</font>'
		else:
			return self.word

	def is_academic(self):
		if self.isAcademic:
			return 1
		else:
			return 0

	def printHtmlCEFR(self):
		if not self.iscefr:
			return self.word
		elif self.cefr == 0:
			return self.word
		else:
			background_color = self.CEFR_bg_color[self.cefr]
			#return self.word + str(self.cefr)
			#return '<span style="background-color: ' + str(background_color) + '">'+ self.word +'</span>'
			#return '<font-stroke><font color="' + str(background_color) + '">'+ self.word +'</font></font-stroke>'
			return '<font color="' + str(background_color) + '">'+ self.word +'</font>'
		

			
class CollectionStats:
	def __init__(self,*args):
		lists = defaultdict(list)
		for arg in args:
			for attr,value in vars(arg).items():
				lists[attr].append(value)

		#average
		self.MeanSentLength = self.average(lists['MeanSentLength'])
		self.num_of_words = self.average(lists['num_of_words'])
		self.MeanWordLength = self.average(lists['MeanWordLength'])
		self.num_stopwords_average = self.average(lists['num_stopwords'])
		self.num_words500 = self.average(lists['num_words500'])
		self.num_words3000 = self.average(lists['num_words3000'])
		self.num_words0 = self.average(lists['num_words0'])
		self.num_academic_average = self.average(lists['num_academic'])
		self.num_unique_academic_average = self.average(lists['num_unique_academic'])
		self.num_of_repetitions = self.average(lists['num_of_repetitions'])
		self.num_linking_average = self.average(lists['num_linking'])
		self.num_collocations_average = self.average(lists['num_collocations'])
		self.num_unique_collocations_average = self.average(lists['num_unique_collocations'])
		self.percVerbs = np.mean(lists['percVerbs'])
		self.totalVerbs = self.average(lists['totalVerbs'])
		self.percInfinitives = np.mean(lists['percInfinitives'])
		self.percParticiples = np.mean(lists['percParticiples'])
		self.percGerunds = np.mean(lists['percGerunds'])
		self.infinitives = self.average(lists['infinitives'])
		self.participles = self.average(lists['participles'])
		self.gerunds = self.average(lists['gerunds'])
		
		#should be absolute counts
		self.NumTexts = len(args)
		self.MaxSentLength = max(lists['MaxSentLength'])
		self.MinSentLength = min(lists['MinSentLength'])
		self.MaxWordLength = max(lists['MaxWordLength'])
		self.MinWordLength = min(lists['MinWordLength'])
		self.num_unclassified = sum(lists['num_unclassified'])
		self.unclassified = ''.join(lists['unclassified']) #list
		self.num_stopwords_total = sum(lists['num_stopwords'])
		self.academic = list(set(itertools.chain(*lists['academic']))) #list
		self.num_academic_total = sum(lists['num_academic'])
		self.num_unique_academic_total = len(self.academic)
		self.num_linking_total = sum(lists['num_linking'])
		self.collocations = list(set(itertools.chain(*[x.keys() for x in lists['collocations']]))) #list
		self.num_collocations_total = sum(lists['num_collocations'])
		self.num_unique_collocations_total = len(self.collocations)
		self.intro_groups = defaultdict(int) #dict
		for k, v in itertools.chain(*[x.items() for x in lists['intro_groups']]):
			self.intro_groups[k] += v
		self.intro_phrase_counter = defaultdict(int) #dict
		for k, v in itertools.chain(*[x.items() for x in lists['intro_phrase_counter']]):
			self.intro_phrase_counter[k] += v
		self.intro_group_dict = defaultdict(list) #dict
		for k, v in itertools.chain(*[x.items() for x in lists['intro_group_dict']]):
			self.intro_group_dict[k] += v
		self.levels = defaultdict(list) #dict
		for k, v in itertools.chain(*[x.items() for x in lists['levels']]):
			self.levels[k] += v
		
	@staticmethod
	def average(vals):
		return np.round(np.mean(vals),2),np.round(np.std(vals),2)
	
	def __str__(self):
		pass #to do levels, average and std
		ResultPage = """<!doctype html>
				<html lang = "en">
								<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
								<script src="/static/engine.js"></script>
						<head>
								<meta charset="utf-8"/>
								<link rel="stylesheet" href="/static/layout.css">
								<title>REALEC Inspector - Inspected</title>
						</head>
						<body onresize="rsz()">
								<div id="h1">REALEC Inspector</div>
								<div id="h1">Statistics for collection</div>
		"""
		ResultPage += """
				<div class="txt" id="cut" style="display: block; font-size: 11pt">"""


		ResultPage += ('<h2>Statistical summary</h2><br>')
		ResultPage += '<b>Number of texts:</b> ' + str(self.NumTexts) + '<br>'
		ResultPage += '<b>Number of words:</b> mean ' + str(self.num_of_words[0]) + ' words, standard deviation ' + str(self.num_of_words[1]) + '.<br>'
		ResultPage += '<b>Average sentence length:</b> mean ' + str(self.MeanSentLength[0]) + ' words, standard deviation ' + str(self.MeanSentLength[1]) + '.<br>'
		ResultPage += '<b>Max sentence length:</b> ' + str(self.MaxSentLength) + ' words.<br>'
		# ResultPage += '<b>Min sentence length:</b> ' + str(self.MinSentLength) + ' words.<br>'
		ResultPage += '<b>Average word length:</b> mean ' + str(self.MeanWordLength[0]) + ' letters, standard deviation ' + str(self.MeanWordLength[1]) + '.<br>'
		ResultPage += '<b>Max word length:</b> ' + str(self.MaxWordLength) + ' letters.<br>'
		# ResultPage += '<b>Min word length:</b> ' + str(self.MinWordLength) + ' letters.<br>'
		'''
		ResultPage += '<b>CEFR</b><br>'
		ResultPage += 'A1: ' + str(len(self.levels['A1'])) + '<br>'
		ResultPage += 'A2: ' + str(len(self.levels['A2'])) + '<br>'
		ResultPage += 'B1: ' + str(len(self.levels['B1'])) + '<br>'
		ResultPage += 'B2: ' + str(len(self.levels['B2'])) + '<br>'
		ResultPage += 'C1: ' + str(len(self.levels['C1'])) + '<br>'
		ResultPage += 'C2: ' + str(len(self.levels['C2'])) + '<br>'
		'''
		ResultPage += 'Unclassified: ' + str(self.num_unclassified) + ' words.<br>'
		ResultPage += 'Stopwords: ' + str(self.num_stopwords_total) + ' total, mean ' + str(self.num_stopwords_average[0]) + ', standard deviation ' + str(self.num_stopwords_average[1]) + '.<br>'
		ResultPage += '<b>Frequency: </b><br>'
		ResultPage += '1-500: mean ' + str(self.num_words500[0]) + ', standard deviation ' + str(self.num_words500[1]) + '.<br>'
		ResultPage += '501-3000: mean ' + str(self.num_words3000[0]) + ', standard deviation ' + str(self.num_words3000[1]) + '.<br>'
		ResultPage += '>3000: mean ' + str(self.num_words0[0]) + ', standard deviation ' + str(self.num_words0[1]) + '.<br>'
		ResultPage += '<b>Academic words</b>: total: ' + str(self.num_academic_total) + ' (' + str(self.num_unique_academic_total) + ' unique), mean: ' + str(self.num_academic_average[0]) + ' (std ' + str(self.num_academic_average[1]) + '), ' + str(self.num_unique_academic_average[0]) + ' are unique (std ' + str(self.num_unique_academic_average[1]) + ').<br>'
		ResultPage += '<b>Word repetitions</b>: mean ' + str(self.num_of_repetitions[0]) + ', standard deviation ' + str(self.num_of_repetitions[1]) + '.<br>'
		ResultPage += '<b>Linking phrases</b>: ' + str(self.num_linking_total) + ' total, mean ' + str(self.num_linking_average[0]) + ', standard deviation ' + str(self.num_linking_average[1]) + '.<br>'
		ResultPage += '<b>Pearsons collocations</b>: total: ' + str(self.num_collocations_total) + ' (' + str(self.num_unique_collocations_total) + ' unique), mean: ' + str(self.num_collocations_average[0]) + ' (std ' + str(self.num_collocations_average[1]) + '), ' + str(self.num_unique_collocations_average[0]) + ' are unique (std ' + str(self.num_unique_collocations_average[1]) + ').<br>'
		'''
		ResultPage += """<br><h2>CEFR</h2><br>"""
		ResultPage += """<img src="data:image/png;base64,%s"/>""" % plot_bar_png(stats)
		ResultPage += '<br>'
		ResultPage += 'A1 level: '
		ResultPage += str(len(self.levels['A1']))
		ResultPage += ' words.<br>'
		ResultPage += str(GetWordsOfLevel(self.levels, 'A1'))
		ResultPage += ('<br><br>A2 level: ')
		ResultPage += str(len(self.levels['A2']))
		ResultPage += (' words.<br>')
		ResultPage += str(GetWordsOfLevel(self.levels, 'A2'))
		ResultPage += ('<br><br>B1 level: ')
		ResultPage += str(len(self.levels['B1']))
		ResultPage += (' words.<br>')
		ResultPage += str(GetWordsOfLevel(self.levels, 'B1'))
		ResultPage += ('<br><br>B2 level: ')
		ResultPage += str(len(self.levels['B2']))
		ResultPage += (' words.<br>')
		ResultPage += str(GetWordsOfLevel(self.levels, 'B2'))
		ResultPage += ('<br><br>C1 level: ')
		ResultPage += str(len(self.levels['C1']))
		ResultPage += (' words.<br>')
		ResultPage += str(GetWordsOfLevel(self.levels, 'C1'))
		ResultPage += ('<br><br>C2 level: ')
		ResultPage += str(len(self.levels['C2']))
		ResultPage += (' words.<br>')
		ResultPage += str(GetWordsOfLevel(self.levels, 'C2'))
		'''
		ResultPage += ('<br><br>Unclassified (0): ')
		ResultPage += str(self.num_unclassified)
		ResultPage += (' words.<br>')
		ResultPage += str(self.unclassified)


		ResultPage += ('<h2>Academic words (Coxhead + COCA)</h2><br>')
		ResultPage += (str(self.num_academic_total)\
				 + ' academic words are in this collection. '\
						+ str(self.num_unique_academic_total) + ' of them are unique.<br>')
		for word in self.academic:
				ResultPage += (word)
				ResultPage += (' ')
		ResultPage += ('<br>')


		ResultPage += ('<h2>Linking Phrases</h2><br>')
		ResultPage += 'There are ' + str(self.num_linking_total) + ' introductory phrases.<br>'
		for elem in self.intro_groups.keys():
				if self.intro_groups[elem] == 0:
						ResultPage += '<i>' + elem + '</i>' + ': 0<br>'
				else:
						ResultPage += '<i>' + elem + '</i>' +': ' + str(self.intro_groups[elem]) + '<br>'
						for phrase in set(self.intro_group_dict[elem]):
								ResultPage += phrase + ': ' + str(self.intro_phrase_counter[phrase]) + '<br>'

		ResultPage += ('<h2>Pearsons Collocations</h2><br>')
		ResultPage += 'There are ' + str(self.num_collocations_total) + ' collocations, ' + \
												str(self.num_unique_collocations_total) + ' of which are unique.<br>'
		for colloc in self.collocations:
				ResultPage += colloc + '; '

		ResultPage += "<br>"
		ResultPage += ('<h2>Verb forms usage</h2><br>')
		ResultPage += '<div class="reddish">Gerunds</div>: mean '+str(self.gerunds[0]) + ', standard deviation ' + str(self.gerunds[1]) +' ('+str(round(self.percGerunds*100,2))+'% of total)'
		ResultPage += '<br><div class="yellowish"><i>to</i> + infinitive forms</div>: mean '+str(self.infinitives[0]) + ', standard deviation ' + str(self.infinitives[1]) +' ('+str(round(self.percInfinitives*100,2))+'% of total)'
		ResultPage += '<br><div class="blueish">Past Participle forms</div>: mean '+str(self.participles[0]) + ', standard deviation ' + str(self.participles[1]) +' ('+str(round(self.percParticiples*100,2))+'% of total)'

#	DEBUG
		'''
		for i in range(len(self.tokens)):
				ResultPage += str(self.tokens[i])+' '
				ResultPage += splitted_text[i]
				ResultPage += "<br>"
		ResultPage += "<br>"+str(len(splitted_text))
		ResultPage += "<br>"+str(len(self.tokens))
		'''		
				
		ResultPage += "</div>"
		

		################# END OF CUT PART ###################


		
		ResultPage += ("""	</body>
</html>""")


		return ResultPage		   



class TextStats:
		def __init__(self,text_type,text_to_inspect):

				self.text = text_to_inspect
				
				self.TextToInspect = text_to_inspect
				
				self.type = text_type
				
				# now = time()
				self.verbsplitted, self.percVerbs, self.totalVerbs, self.percInfinitives, self.percParticiples, \
				self.percGerunds, self.infinitives, self.participles, self.gerunds, tagged = inspect_verbs(self.text)
				# print('total',time()-now)
				
				parsed_text = get_parsed_text(self.text)

				self.text = re.sub(r'[^\x00-\x7F]+','', self.text)

				self.clean_words_list, self.stats, self.coca_freq, self.academic, self.num_of_words, self.levels, \
				self.overall_counter, self.intro_group_counter, self.intro_group_dict, self.intro_group_list, self.intro_phrase_counter, \
				self.intro_groups, self.tokens, self.collocations, self.stopwords_in_text, self.time_and_sequence = GetStats(self.text)
				
				self.ff = DealWithFreq(self.coca_freq)
				self.num_of_repetitions = sum(self.overall_counter.values()) - len(self.overall_counter)
				self.most_common_repetition = self.overall_counter.most_common(1)[0]

				self.MeanSentLength = GetMeanSentLength(self.text)
				self.MaxSentLength = GetMaxSentLength(self.text)
				self.MinSentLength = GetMinSentLength(self.text)
				self.MeanWordLength = GetMeanWordLength(self.text)
				self.MaxWordLength = GetMaxWordLength(self.text)
				self.MinWordLength = GetMinWordLength(self.text)

				self.num_unclassified = len(self.levels['0'])
				self.num_stopwords = len(self.levels['stopwords'])
				self.num_words500 = self.ff.GetNum('1-500')
				self.num_words3000 = self.ff.GetNum('501-3000')
				self.num_words0 = self.ff.GetNum('>3000')
				self.num_academic = len(self.academic)
				self.num_unique_academic = len(set(self.academic))
				self.num_linking = sum(self.intro_groups.values())
				self.num_collocations = sum(self.collocations.values())
				self.num_unique_collocations = len(self.collocations.keys())

				self.unclassified = GetWordsOfLevel(self.levels, '0')
				self.stopwords = GetWordsOfLevel(self.levels, 'stopwords')
				
				self.words500 = self.ff.GetWordsOfThatLevel('1-500')
				self.words3000 = self.ff.GetWordsOfThatLevel('501-3000')
				self.words0 = self.ff.GetWordsOfThatLevel('>3000')
				
				if self.type == "opinion":
					self.num_tokens = count_tokens(parsed_text)
					self.W = self.num_tokens
					ms_q, self.list_spell_q = run_enchant(self.TextToInspect)
					self.misspelled_q = ms_q / float(self.num_of_words)
					self.E = self.misspelled_q / float(self.num_of_words)
					self.L = self.MeanWordLength
					self.V = self.percVerbs
					self.uniques_quan, self.list_unique = CountUniques(tagged)
					self.U = self.uniques_quan / float(self.num_of_words)
					self.R = self.most_common_repetition[1] / float(self.num_of_words)
					self.B1 = len(self.levels['B1']) / float(self.num_of_words)
					self.T = self.intro_groups['Time and sequence'] / float(self.num_of_words)
					self.S = self.MaxSentLength
					self.num_sents = count_sent(parsed_text)
					self.N = self.num_sents / float(self.num_tokens)
					self.C1 = len(self.levels['C1']) / float(self.num_of_words)
					self.X = self.MaxWordLength
					self.ger_inf, self.list_ger_inf = find_ger_inf(parsed_text)
					self.GI = self.ger_inf / float(self.num_of_words)
					
					self.M = 44.111 + 0.035*self.W - 62.392*self.E + 5.065*self.L + 38.232*self.V - 28.477*self.U - 84.454*self.R + 43.072*self.B1 - 50.749*self.T - 0.106*self.S - 86.793*self.N + 69.109*self.C1 + 0.18*self.X + 24.618*self.GI
					
					self.ValuesDict = { \
					'type': 'opinion', \
					'predicted_mark': self.M, \
					'num_tokens': self.W, \
					'spell_q': self.E, \
					'MeanWordLength': self.L, \
					'verbs_quan': self.V, \
					'num_unique': self.U, \
					'num_most_common_repetition': self.R, \
					'num_b1': self.B1, \
					'TimeAndSequence': self.T, \
					'MaxSentLength': self.S, \
					'num_sents': self.N, \
					'num_c1': self.C1, \
					'MaxWordLength': self.X, \
					'ger_inf': self.GI \
					}
					
				if self.type == "graph":
					self.num_tokens = count_tokens(parsed_text)
					self.W = self.num_tokens
					self.W0 = len(self.levels['0']) / float(self.num_of_words)
					self.L = self.MeanWordLength
					self.num_sents = count_sent(parsed_text)
					self.S = self.num_sents / float(self.num_tokens)
					self.NS = self.num_stopwords / float(self.num_of_words)
					self.num_intro_groups = sum(self.intro_groups.values())
					self.I = self.num_intro_groups / float(self.num_of_words)
					self.SL = self.MaxSentLength
					self.num_np, self.list_np = count_np(parsed_text)
					self.NP = self.num_np / float(self.num_tokens)
					self.num_cl = count_clauses(parsed_text)
					self.C = self.num_cl / float(self.num_tokens)
					self.V = self.percVerbs
					
					self.M = 70.453 + 0.047*self.W - 56.568*self.W0 + 2.983*self.L - 106.765*self.S - 43.062*self.NS + 41.667*self.I - 0.08*self.SL - 31.861*self.NP - 53.744*self.C + 36.876*self.V
					
					self.ValuesDict = { \
					'type': 'graph', \
					'predicted_mark': self.M, \
					'num_tokens': self.W, \
					'num_0': self.W0, \
					'MeanWordLength': self.L, \
					'num_sents': self.S, \
					'num_stopwords': self.NS, \
					'num_intro_groups': self.I, \
					'MaxSentLength': self.SL, \
					'num_np': self.NP, \
					'num_cl': self.C, \
					'verbs_quan': self.V \
					}

		def __str__(self):
				if self.type == 'graph':
					text_type_str = 'graph description'
				else:
					text_type_str = 'opinion essay'
				
				presplit = self.text.split()
				delarr=[]
				for w in range(len(presplit)):
					if not any(ord(char) < 128 for char in presplit[w]):
							delarr.append(w)
				for index in delarr:
						del presplit[index]
				
				if len(presplit) < 150:
						sleep(1)
						ShorterError = """<!doctype html>
								<html lang = "en">
								<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
								<script src="/static/engine.js"></script>
						<head>
								<meta charset="utf-8"/>
								<link rel="stylesheet" href="/static/layout.css">
								<title>REALEC Inspector - Text too short</title>

						</head>
						<body onload="fadein(document.body,500)">
								<div id="h1">REALEC Inspector</div>
								<div class="txt" style="width: 80%; margin: auto; text-align: center; font-size: 13pt; color: #CC0000;">Your text has <b>"""
						ShorterError += str(len(presplit))
						ShorterError += """</b> words. The current standard requires that a text in should have not less than 150 words. Please go back to your """
						ShorterError += text_type_str
						ShorterError += """ and complete the text.</div>
								</body>
								</html>"""

						return ShorterError
						# "Seems like your text contains", len(text_to_inspect.split()),"words. Given the current IELTS standarts,\
						# any text with less than 150 words should be evaluated with 0 points. Please revert to the previous page and complement your text"

				html_text = self.text.replace('\r\n', '<br>')
				html_text = html_text.replace('\n', '<br>')
				html_text = html_text.replace('\n', '<br>')

				splitted_text = re.sub(r'<br>',r'<br> ',html_text).split()
				
				
				ResultPage = """<!doctype html>
		<html lang = "en">
						<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
						<script src="/static/engine.js"></script>
				<head>
						<meta charset="utf-8"/>
						<link rel="stylesheet" href="/static/layout.css">
						<title>REALEC Inspector - Inspected</title>
				</head>
				<body onresize="rsz()">
						<div id="h1">REALEC Inspector</div>
						<div class="txt" style="width: 80%; margin: auto; text-align: center; font-size: 13pt; font-weight: bold;">Here is your """+text_type_str+""":</div>
						<div class="txt" id="hiddenplacer" style="position: relative; width: 98%; margin: auto; text-align: center; z-index: 0; visibility: hidden;">"""
		#	ResultPage += """<h2>Source Essay</h2><br>"""
		#	with open ("/home/ivan/outt.txt","w",encoding="utf-8") as t:
		#		t.write(text_to_inspect)
				
				x = re.sub(r'\n','<br>',self.TextToInspect)
				ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)

				ResultPage += """<div class="txt" id="textplacer" style="position: absolute; width: 100%; top: 0px; left: 0px; text-align: center; visibility: visible; z-index: 0;">"""
				x = re.sub(r'\n','<br>',self.TextToInspect)
				ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)

				'''
				ResultPage += ('<h2>Visualization</h2>')
				
				ResultPage += ('<h3>CEFR</h3>')


				a1_words, a2_words, b1_words, b2_words, c1_words, c2_words = str(), str(), str(), str(), str(), str()

				k=-1
				for i in range(len(splitted_text)):
						k=k+1
						if re.search(r'[a-zA-Z0-9]',splitted_text[i]) == None:
								k=k-1
								continue
						else:
								a1_words += textcolor(splitted_text[i],self.tokens[k].is_cefr('A1')) + ' '
								a2_words += textcolor(splitted_text[i],self.tokens[k].is_cefr('A2')) + ' '
								b1_words += textcolor(splitted_text[i],self.tokens[k].is_cefr('B1')) + ' '
								b2_words += textcolor(splitted_text[i],self.tokens[k].is_cefr('B2')) + ' '
								c1_words += textcolor(splitted_text[i],self.tokens[k].is_cefr('C1')) + ' '
								c2_words += textcolor(splitted_text[i],self.tokens[k].is_cefr('C2')) + ' '

				ResultPage += input_toggle('Click to show A1-level words in text', a1_words)
				ResultPage += input_toggle('Click to show A2-level words in text', a2_words)
				ResultPage += input_toggle('Click to show B1-level words in text', b1_words)
				ResultPage += input_toggle('Click to show B2-level words in text', b2_words)
				ResultPage += input_toggle('Click to show C1-level words in text', c1_words)
				ResultPage += input_toggle('Click to show C2-level words in text', c2_words)
				'''



				ResultPage += """</div>"""
				
				
				################# BEGINNING OF VISUALISATION PART ###################


		#	ResultPage += ('<br><br><h3>Frequency</h3>')

				words500, words3000, words0 = str(), str(), str()
				k=-1
				for i in range(len(splitted_text)):
						k=k+1
						if not any(ord(char) < 128 for char in splitted_text[i]):
								words500 += splitted_text[i] + ' '
								words3000 += splitted_text[i] + ' '
								words0 += splitted_text[i] + ' '
								k=k-1
		#			continue
						elif splitted_text[i] == '<br>':
								k=k-1
								words500 = words500[:-1]+splitted_text[i]
								words3000 = words3000[:-1]+splitted_text[i]
								words0 = words0[:-1]+splitted_text[i]
		#			continue
						else:
								words500 += textcolor(splitted_text[i],Blue,self.tokens[k].is_freq('1-500')) + ' '
								words3000 += textcolor(splitted_text[i],Blue,self.tokens[k].is_freq('501-3000')) + ' '
								words0 += textcolor(splitted_text[i],Red,self.tokens[k].is_freq(0)) + ' '

				ResultPage += """<div class="txt" id="words_500" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
				ResultPage += re.sub(r'( )*<br>( )*',r'<br>',words500)
				ResultPage += """</div>
						<div class="txt" id="words_3000" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
				ResultPage += re.sub(r'( )*<br>( )*',r'<br>',words3000)
				ResultPage += """</div>
						<div class="txt" id="words_0" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
				ResultPage += re.sub(r'( )*<br>( )*',r'<br>',words0)


		#	ResultPage += ('<br><br><h3>Academic Words</h3>')

				
				ResultPage += """</div>
						<div class="txt" id="verbs" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
				x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.verbsplitted))
				ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
				
				if self.type == 'graph':

					self.list_num_0 = []
					for token in self.tokens:
						if token.cefr == "0":
							self.list_num_0.append((token.word,Red))
						else:
							self.list_num_0.append((token.word,None))
					
					self.list_stopwords = []
					for token in self.tokens:
						if token.isStopword:
							self.list_stopwords.append((token.word,Red))
						else:
							self.list_stopwords.append((token.word,None))
					
					self.list_intro_groups = [(group, Green) for group in self.intro_group_list]
					
				else:
					
					self.list_most_common_repetition = [(self.most_common_repetition[0],Red) for i in range(self.most_common_repetition[1])]
					
					self.list_b1 = []
					for token in self.tokens:
						if token.cefr == "B1":
							self.list_b1.append((token.word,Green))
						else:
							self.list_b1.append((token.word,None))
					
					self.list_TimeAndSequence = [(item, Red) for item in self.time_and_sequence]
					
					self.list_c1 = []
					for token in self.tokens:
						if token.cefr == "C1":
							self.list_c1.append((token.word,Green))
						else:
							self.list_c1.append((token.word,None))
				
				if self.type == 'graph':
					ResultPage += """</div>
							<div class="txt" id="num_0" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_num_0))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
					ResultPage += """</div>
							<div class="txt" id="stopwords" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_stopwords))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
					ResultPage += """</div>
							<div class="txt" id="intro_groups" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_intro_groups))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
					ResultPage += """</div>
							<div class="txt" id="np" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_np))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
				
				else:
					ResultPage += """</div>
							<div class="txt" id="spell_q" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_spell_q))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
					ResultPage += """</div>
							<div class="txt" id="unique" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_unique))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
					ResultPage += """</div>
							<div class="txt" id="most_common_repetition" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_most_common_repetition))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
					ResultPage += """</div>
							<div class="txt" id="b1" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_b1))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
					ResultPage += """</div>
							<div class="txt" id="TimeAndSequence" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_TimeAndSequence))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
					ResultPage += """</div>
							<div class="txt" id="c1" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_c1))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
					ResultPage += """</div>
							<div class="txt" id="ger_inf" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
					x = re.sub(r'\n','<br>',colorize_by_list(self.TextToInspect, self.list_ger_inf))
					ResultPage += re.sub(r'( )*<br>( )*',r'<br>',x)
							
				ResultPage += """</div>"""
				
		#	ResultPage += str(splitted_text)

		#	ResultPage += ('<br><br><h2>What May Be Improved</h2><br>')


				################# HOLY END ###################


				ResultPage += """</div>"""	




				################# EVALUATION PART #########################
				
				ResultPage += """<br>
						<br>
						<div class="txt" style="width: 80%; margin: auto; text-align: center; font-size: 13pt; font-weight: bold;">Here is how it compares with the other preparational essays:</div>
						<table border="0" cellspacing="3" class="txt" style="margin-left: 2%">"""
				
				ResultPage += GenerateEvaluation(self.ValuesDict)
				
				################# END OF EVALUATION PART ##################


				ResultPage += """		</table>
						<div class="txt" id="showcut" style="width: 80%; margin: auto; text-align: center; font-size: 9pt;  display: block;"><a href="javascript:showcut()">click to show advanced statistical data &#8595;</a></div>
						<div class="txt" id="hidecut" style="width: 80%; margin: auto; text-align: center; font-size: 9pt; opacity: 0; display: none;"><a href="javascript:hidecut()">hide advanced data &#8593;</a></div>"""


				################# NEATY LITTLE POPUPBOXES ##################

				ResultPage += """<div class="txt" name="popupbox" id="tokens" style="position: absolute; width: auto; top: 0px; left: 0px; font-size: 10px; background-color: white; border: 3px dashed; border-color: rgb(190,16,123); border-radius: 5px; opacity: 0; display: none;">"""
				ResultPage += str(self.num_of_words)
				ResultPage += """ words</div>
				"""
				ResultPage += """<div class="txt" name="popupbox" id="MeanWordLength" style="position: absolute; width: auto; top: 0px; left: 0px; font-size: 10px; background-color: white; border: 3px dashed; border-color: rgb(190,16,123); border-radius: 5px; opacity: 0; display: none;">"""
				ResultPage += str(round(self.MeanWordLength,2))
				ResultPage += """ symbols</div>
				"""
				ResultPage += """<div class="txt" name="popupbox" id="sents" style="position: absolute; width: auto; top: 0px; left: 0px; font-size: 10px; background-color: white; border: 3px dashed; border-color: rgb(190,16,123); border-radius: 5px; opacity: 0; display: none;">"""
				if self.type == "graph":
					ResultPage += str(round(self.S,2))
				else:
					ResultPage += str(round(self.N,2))
				ResultPage += """ sentences/words</div>
				"""
				ResultPage += """<div class="txt" name="popupbox" id="MaxSentLength" style="position: absolute; width: auto; top: 0px; left: 0px; font-size: 10px; background-color: white; border: 3px dashed; border-color: rgb(190,16,123); border-radius: 5px; opacity: 0; display: none;">"""
				ResultPage += str(self.MaxSentLength)
				ResultPage += """ words</div>
				"""
				
				if self.type == 'graph':
					ResultPage += """<div class="txt" name="popupbox" id="cl" style="position: absolute; width: auto; top: 0px; left: 0px; font-size: 10px; background-color: white; border: 3px dashed; border-color: rgb(190,16,123); border-radius: 5px; opacity: 0; display: none;">"""
					ResultPage += str(round(self.C,2))
					ResultPage += """ clauses/words</div>
					"""
				
				else:
					ResultPage += """<div class="txt" name="popupbox" id="MaxWordLength" style="position: absolute; width: auto; top: 0px; left: 0px; font-size: 10px; background-color: white; border: 3px dashed; border-color: rgb(190,16,123); border-radius: 5px; opacity: 0; display: none;">"""
					ResultPage += str(self.MaxWordLength)
					ResultPage += """ symbols</div>
					"""
				
				################# POPUPBOXES END ###################







				################# BEGINNING OF CUT PART ###################


				ResultPage += """
						<div class="txt" id="cut" style="display: block; font-size: 11pt; display: none;">"""
				 
				ResultPage += '<div style="display: inline-block; text-align: center;">Mark predicted by REALEC Inspector: <b>'
				ResultPage += str(round(self.M,2))
				ResultPage += '</b></div>'
				
				ResultPage += ('<h2>Statistical summary</h2><br>')
				ResultPage += '<b>Number of words:</b> ' + str(self.num_of_words) + '<br>'
				ResultPage += '<b>Average sentence length:</b> ' + str(self.MeanSentLength) + ' words.<br>'
				ResultPage += '<b>Max sentence length:</b> ' + str(self.MaxSentLength) + ' words.<br>'
				# ResultPage += '<b>Min sentence length:</b> ' + str(self.MinSentLength) + ' words.<br>'
				ResultPage += '<b>Average word length:</b> ' + str(self.MeanWordLength) + ' letters.<br>'
				ResultPage += '<b>Max word length:</b> ' + str(self.MaxWordLength) + ' letters.<br>'
				# ResultPage += '<b>Min word length:</b> ' + str(self.MinWordLength) + ' letters.<br>'
				'''
				ResultPage += '<b>CEFR</b><br>'
				ResultPage += 'A1: ' + str(len(self.levels['A1'])) + '<br>'
				ResultPage += 'A2: ' + str(len(self.levels['A2'])) + '<br>'
				ResultPage += 'B1: ' + str(len(self.levels['B1'])) + '<br>'
				ResultPage += 'B2: ' + str(len(self.levels['B2'])) + '<br>'
				ResultPage += 'C1: ' + str(len(self.levels['C1'])) + '<br>'
				ResultPage += 'C2: ' + str(len(self.levels['C2'])) + '<br>'
				'''
				ResultPage += 'Unclassified: ' + str(self.num_unclassified) + '<br>'
				ResultPage += 'Stopwords: ' + str(self.num_stopwords) + '<br>'
				ResultPage += '<b>Frequency: </b><br>'
				ResultPage += '1-500: ' + str(self.num_words500) + '<br>'
				ResultPage += '501-3000: ' + str(self.num_words3000) + '<br>'
				ResultPage += '>3000: ' + str(self.num_words0) + '<br>'
				ResultPage += '<b>Academic words</b>: ' + str(self.num_academic) + ' (' + str(self.num_unique_academic) + ' unique)<br>'
				ResultPage += '<b>Word repetitions</b>: ' + str(self.num_of_repetitions) + \
												' (' + str(self.most_common_repetition) + ' is the most repeated) <br>'
				ResultPage += '<b>Linking phrases</b>: ' + str(self.num_linking) + '<br>'
				ResultPage += '<b>Pearsons collocations</b>: ' + str(self.num_collocations) + ' (' +\
														str(self.num_unique_collocations) + ' unique)<br>'
				'''
				ResultPage += """<br><h2>CEFR</h2><br>"""
				ResultPage += """<img src="data:image/png;base64,%s"/>""" % plot_bar_png(stats)
				ResultPage += '<br>'
				ResultPage += 'A1 level: '
				ResultPage += str(len(self.levels['A1']))
				ResultPage += ' words.<br>'
				ResultPage += str(GetWordsOfLevel(self.levels, 'A1'))
				ResultPage += ('<br><br>A2 level: ')
				ResultPage += str(len(self.levels['A2']))
				ResultPage += (' words.<br>')
				ResultPage += str(GetWordsOfLevel(self.levels, 'A2'))
				ResultPage += ('<br><br>B1 level: ')
				ResultPage += str(len(self.levels['B1']))
				ResultPage += (' words.<br>')
				ResultPage += str(GetWordsOfLevel(self.levels, 'B1'))
				ResultPage += ('<br><br>B2 level: ')
				ResultPage += str(len(self.levels['B2']))
				ResultPage += (' words.<br>')
				ResultPage += str(GetWordsOfLevel(self.levels, 'B2'))
				ResultPage += ('<br><br>C1 level: ')
				ResultPage += str(len(self.levels['C1']))
				ResultPage += (' words.<br>')
				ResultPage += str(GetWordsOfLevel(self.levels, 'C1'))
				ResultPage += ('<br><br>C2 level: ')
				ResultPage += str(len(self.levels['C2']))
				ResultPage += (' words.<br>')
				ResultPage += str(GetWordsOfLevel(self.levels, 'C2'))
				'''
				ResultPage += ('<br><br>Unclassified (0): ')
				ResultPage += str(self.num_unclassified)
				ResultPage += (' words.<br>')
				ResultPage += str(self.unclassified)
				ResultPage += ('<br><br>Stopwords: ')
				ResultPage += str(self.num_stopwords)
				ResultPage += (' words.<br>')
				ResultPage += str(self.stopwords)
				ResultPage += ('<br>')

				ResultPage += ('<h2>Frequency Stats</h2><br>')
				ResultPage += """<a class="innerlink" id="words_500link" onclick="reversemaskerlink(this,'words_500')" href="javascript:showmasker('words_500')">1-500</a>: """
				ResultPage += str(self.num_words500)
				ResultPage += (' words.<br>')
				ResultPage += str(self.words500)
				ResultPage += """<br><br><a class="innerlink" id="words_3000link" onclick="reversemaskerlink(this,'words_3000')" href="javascript:showmasker('words_3000')">501-3000</a>: """
				ResultPage += str(self.num_words3000)
				ResultPage += (' words.<br>')
				ResultPage += str(self.words3000)
				ResultPage += """<br><br><a class="innerlink" id="words_0link" onclick="reversemaskerlink(this,'words_0')" href="javascript:showmasker('words_0')">>3000</a>: """
				ResultPage += str(self.num_words0)
				ResultPage += (' words.<br>')
				ResultPage += str(self.words0)

				ResultPage += ('<h2>Academic words (Coxhead + COCA)</h2><br>')
				ResultPage += (str(self.num_academic)\
						 + ' academic words are in this text. '\
								+ str(self.num_unique_academic) + ' of them are uniqe.<br>')
				for word in set(self.academic):
						ResultPage += (word)
						ResultPage += (' ')
				ResultPage += ('<br>')


				ResultPage += ('<h2>Distribution Graphs</h2><br>')
				ResultPage += (plot_distr(mean_sent_lengths, self.MeanSentLength))
				ResultPage += ('<br>')
				ResultPage += (getPerc(mean_sent_lengths, self.MeanSentLength))
				ResultPage += ('% of texts in the realec corpus have mean sentence length less than in the source text.<br>')

				
				ResultPage += (plot_distr(mean_word_lengths, self.MeanWordLength, 'Mean word length', 0.1))
				ResultPage += ('<br>')
				ResultPage += (getPerc(mean_word_lengths, self.MeanWordLength))
				ResultPage += ('% of texts in the realec corpus have mean word length less than in the source text.<br>')


				ResultPage += ('<h2>Word Repetitions</h2><br>')
				ResultPage += 'Overall there are ' + str(self.num_of_repetitions) + ''' word repetitions in this text.\
										The most common of them are:<br>'''
				for elem in self.overall_counter.most_common(5):
						ResultPage += str(elem[0])
						ResultPage += ': '
						ResultPage += str(elem[1])
						ResultPage += ' times <br>'


				ResultPage += ('<h2>Linking Phrases</h2><br>')
				ResultPage += 'There are ' + str(self.num_linking) + ' introductory phrases.<br>'
				for elem in self.intro_groups.keys():
						if self.intro_groups[elem] == 0:
								ResultPage += '<i>' + elem + '</i>' + ': 0<br>'
						else:
								ResultPage += '<i>' + elem + '</i>' +': ' + str(self.intro_groups[elem]) + '<br>'
								for phrase in set(self.intro_group_dict[elem]):
										ResultPage += phrase + ': ' + str(self.intro_phrase_counter[phrase]) + '<br>'

				ResultPage += ('<h2>Pearsons Collocations</h2><br>')
				ResultPage += 'There are ' + str(self.num_collocations) + ' collocations, ' + \
														str(self.num_unique_collocations) + ' of which are unique.<br>'
				for colloc in self.collocations.keys():
						ResultPage += colloc + '; '

				ResultPage += "<br>"
				ResultPage += ('<h2>Verb forms usage</h2><br>')
				ResultPage += '<div class="reddish">Gerunds</div>: '+str(self.gerunds)+' ('+str(round(self.percGerunds*100,2))+'% of total)'
				ResultPage += '<br><div class="yellowish"><i>to</i> + infinitive forms</div>: '+str(self.infinitives)+' ('+str(round(self.percInfinitives*100,2))+'% of total)'
				ResultPage += '<br><div class="blueish">Past Participle forms</div>: '+str(self.participles)+' ('+str(round(self.percParticiples*100,2))+'% of total)'

		#	DEBUG
				'''
				for i in range(len(self.tokens)):
						ResultPage += str(self.tokens[i])+' '
						ResultPage += splitted_text[i]
						ResultPage += "<br>"
				ResultPage += "<br>"+str(len(splitted_text))
				ResultPage += "<br>"+str(len(self.tokens))
				'''		
						
				ResultPage += "</div>"
				

				################# END OF CUT PART ###################


				
				ResultPage += ("""	</body>
		</html>""")


				return ResultPage			

				

def textcolor(word,color,colorize):
	wordx = re.sub("<br>","%&%&",word)
	if colorize:
		retstr = re.sub(r'([a-zA-Z]+)',r'<font color="'+color+r'">\1</font>',wordx)
		retstr = re.sub("%&%&","<br>",retstr)
		return retstr
	else:
		wordx = re.sub("%&%&","<br>",wordx)
		return wordx
#		return '<font color="#222222">'+word+'</font>'

def colorize_by_list(intext, tupleslist):
	colorized = ''
	incopy = intext
	for intuple in tupleslist:
		# print(intuple[0])
		x = re.search('(\W|^)('+re.escape(intuple[0])+')(\W|$s)', incopy, flags=re.IGNORECASE)
		if x:
			colorized += incopy[:x.span(2)[0]]
			if intuple[1] is not None:
				colorized += """<div style="display: inline; color: """
				colorized += intuple[1]
				colorized += '">'
				colorized += x.group(2)
				colorized += '</div>'
			else:
				colorized += x.group(2)
			incopy = incopy[x.span(2)[1]:]
	colorized += incopy
	return colorized

def GetStats(text_to_inspect, is_wff = True):
	eng_voc_stats = Counter({'A1' : 0,'A2' : 0,'B1' : 0,'B2' : 0,'C1' : 0,'C2' : 0, 'stopwords' : 0, '0': 0})
	wff_stats = Counter({'A1' : 0,'A2' : 0,'B1' : 0,'B2' : 0,'C1' : 0,'C2' : 0, 'stopwords' : 0, '0': 0})

	def GetLemmaLevel(word):
		word_noun = wordnet_lemmatizer.lemmatize(word, pos = 'n')
		word_verb = wordnet_lemmatizer.lemmatize(word, pos = 'v')
		word_adj = wordnet_lemmatizer.lemmatize(word, pos = 'a')
		levels = (wff[word], wff[word_noun], wff[word_verb], wff[word_adj])
		for level in levels:
			if level != 0:
				return level
		return '0'

	intro_groups = Counter({'Time and sequence' : 0,
		  'Comparison' : 0,
		  'Contrast' : 0,
		  'Examples' : 0,
		  'Cause and Effect' : 0,
		  'Giving reasons, explanations' : 0,
		  'Addition' : 0,
		  'Concession' : 0,
		  'Conclusion and summary' : 0,
		  'Repetition' : 0})

	list_of_words = text_to_inspect.split()

	clean_words_list = []
	coca_freq = Counter()
	levels_wff = defaultdict(list)
	levels_ev = defaultdict(list)
	unique_words = []
	academic = []
	stopwords_in_text = []
	overall_counter = Counter() # schitaet skolko raz voshlo kazhdoe slovo ne is stoplista
	intro_phrase_counter = Counter()
	intro_group_counter = Counter()
	intro_group_dict = defaultdict(list)
	intro_group_list = []
	time_and_sequence = []
	tokens = []
	collocations = Counter()

	for word in list_of_words:
		clean_word = word.strip().lower().translate(translator)
		clean_words_list.append(clean_word)

		word_token = Token(clean_word)
		
		if (clean_word not in stopwords_eng) and not isNum(clean_word):
			# eng_voc_stats[eng_voc[clean_word]] += 1
			# wff_stats[wff[clean_word]] += 1
			wff_stats[GetLemmaLevel(clean_word)] += 1
			overall_counter[clean_word] += 1
			# word_token.setTokenCEFR(wff[clean_word])
			word_token.setTokenCEFR(GetLemmaLevel(clean_word))
		else:
			eng_voc_stats['stopwords'] += 1
			wff_stats['stopwords'] += 1
			stopwords_in_text.append(clean_word)
			word_token.setTokenStopword()

		if (clean_word not in stopwords_eng) and not isNum(clean_word):
			coca_freq[clean_word] = COCA_freq[clean_word]
			word_token.setTokenFreq(COCA_freq[clean_word])
		
		tokens.append(word_token)

		if (clean_word in awl or clean_word in COCA_academic) and clean_word not in stopwords_eng:
			academic.append(clean_word)
			word_token.setTokenAcademic(True)

		if clean_word not in unique_words:
			unique_words.append(clean_word)

		'''
		if clean_word in introwords:
			intro_phrase_counter[clean_word] += 1
			group = introwords_hash[clean_word]
			intro_group_counter[group] += 1
			intro_group_dict[group].append(clean_word)
			intro_groups[group] += 1
		'''
	num_of_words = len(clean_words_list)

	for i in range(len(list_of_words) - 2):
		word = list_of_words[i].strip().lower().translate(translator)
		next_word = list_of_words[i+1].strip().lower().translate(translator)
		next_next_word = list_of_words[i+2].strip().lower().translate(translator)
		if pearson_hash_table[word] != 0:
			add = False
			if (next_word in pearson_hash_table[word]):
				colloc = word + ' ' + next_word
				add = True
			if ((next_word + ' ' + next_next_word) in pearson_hash_table[word]):
				colloc = word + ' ' + next_word + ' ' + next_next_word
				add = True
			if add:
				collocations[colloc] += 1
				add = False
		if word in introwords:
			intro_phrase_counter[word] += 1
			group = introwords_hash[word]
			intro_group_counter[group] += 1
			intro_group_dict[group].append(word)
			intro_group_list.append(word)
			intro_groups[group] += 1
			if group == "Time and sequence":
				time_and_sequence.append(word)
		elif word + ' ' + next_word in introwords:
			phrase = word + ' ' + next_word
			intro_phrase_counter[phrase] += 1
			group = introwords_hash[phrase]
			intro_group_counter[group] += 1
			intro_group_dict[group].append(phrase)
			intro_group_list.append(phrase)
			intro_groups[group] += 1
			if group == "Time and sequence":
				time_and_sequence.append(phrase)
	word = list_of_words[-2].strip().lower().translate(translator)
	next_word = list_of_words[-1].strip().lower().translate(translator)
	if pearson_hash_table[word] != 0:
		if (next_word in pearson_hash_table[word]):
			colloc = word + ' ' + next_word
			collocations[colloc] += 1

	for word in unique_words:
		if word not in stopwords_eng:
			# levels_wff[wff[word]].append(word)
			levels_wff[GetLemmaLevel(word)].append(word)
			levels_ev[eng_voc[word]].append(word)
		else:
			levels_wff['stopwords'].append(word)
			levels_ev['stopwords'].append(word)

	if is_wff == True:
		return clean_words_list, wff_stats, coca_freq, academic, num_of_words, levels_wff, overall_counter,\
				intro_group_counter, intro_group_dict, intro_group_list, intro_phrase_counter, intro_groups, tokens, collocations, stopwords_in_text, time_and_sequence
	else:
		return clean_words_list, eng_voc_stats, coca_freq, academic, num_of_words, levels_ev, overall_counter,\
				intro_group_counter, intro_group_dict, intro_group_list, intro_phrase_counter, intro_groups, tokens, collocations, stopwords_in_text, time_and_sequence

def GetWordsOfLevel(levels_dict, level):
	resp = ''
	if len(levels_dict[level]) != 0:
		for word in levels_dict[level]:
			resp += word + ', '
		return resp
	else:
		return 'There are no words of that level in this text.'


class DealWithFreq(object):

	def __init__(self, coca):
		self.levels_content = defaultdict(list)
		for word, freq in coca.items():
			if freq == '1-500':
				self.levels_content['1-500'].append(word)
			elif freq == '501-3000':
				self.levels_content['501-3000'].append(word)
			else:
				self.levels_content['>3000'].append(word)

	def __init__(self, coca):
		self.levels_content = defaultdict(list)
		for word, freq in coca.items():
			if freq == '1-500':
				self.levels_content['1-500'].append(word)
			elif freq == '501-3000':
				self.levels_content['501-3000'].append(word)
			else:
				self.levels_content['>3000'].append(word)

	def GetWordsOfThatLevel(self, level):
		resp = ''
		for word in self.levels_content[level]:
			resp += word + ', '
		return resp

	def GetNum(self, level):
		return len(self.levels_content[level])

	def GetWordsOfThatLevel(self, level):
		resp = ''
		for word in self.levels_content[level]:
			resp += word + ', '
		return resp

	def GetNum(self, level):
		return len(self.levels_content[level])

def TestNL(t):
	text = nltk.word_tokenize(t)
	return nltk.pos_tag(text)[0][1]

def GetMeanSentLength(textik):
	lengths = []
	for sent in textik.replace('?','.').replace('!','.').split('.'):
		lengths.append(len(sent.split(' ')))
	# print lengths
	return float(sum(lengths)) / len(lengths)

def GetMeanWordLength(textik):
	lengths = []
	for word in textik.translate(translator).split(' '):
		lengths.append(len(word))
	return float(sum(lengths)) / len(lengths)

def GetMaxSentLength(textik):
	max_len = 0
	for sent in textik.replace('?','.').replace('!','.').split('.'):
		if len(sent.split(' ')) > max_len:
			max_len = len(sent.split(' '))
	return max_len

def GetMinSentLength(textik):
	min_len = 999
	for sent in textik.replace('?','.').replace('!','.').split('.'):
		if len(sent.split(' ')) < min_len:
			min_len = len(sent.split(' '))
	return min_len

def GetMaxWordLength(textik):
	max_len = 0
	for word in textik.translate(translator).split(' '):
		if len(word) > max_len:
			max_len = len(word)
	return max_len

def GetMinWordLength(textik):
	min_len = 999
	for word in textik.translate(translator).split(' '):
		if 0 < len(word) < min_len:
			min_len = len(word)
	return min_len

def getPerc(array, element, string = 1):
	k = 0
	for el in array:
		if el < element:
			k += 1
	str_repr = str(float(k) / len(array) * 100)
	if string == 0:
		return str(float(k) / len(array) * 100)
	if len(str_repr) > 4:
		return str_repr[:4]
	return str_repr

def IsCEFRRec(stats):
	if stats['C2'] < 1 or stats['C1'] < 2 or stats['B2'] < 8 or stats['B1'] < 16:
		return True

def plot_bar_png(stats):
	filename = 'bar.png'
	percent = []
	keys = []
	num_of_words = sum(stats.values())
	for val in sorted(stats.items(), key=itemgetter(0)):
		percent.append(val[1] / float(num_of_words) * 100)
		keys.append(val[0])

	fig = plt.figure()  # Plotting
	canvas = FigureCanvas(fig)

	plt.title("CEFR Distribution")
	barlist= plt.bar(range(len(stats)), percent, align='center')
	plt.xticks(range(len(stats)), keys)
	barlist[0].set_color('k')
	barlist[1].set_color('#FF4500')
	barlist[2].set_color('#5F9EA0')
	barlist[3].set_color('#8B0000')
	barlist[4].set_color('#228B22')
	barlist[5].set_color('#DAA520')
	barlist[6].set_color('#00BFFF')
	barlist[7].set_color('k')

	plt.ylabel('%')


	v_io = io.BytesIO()
	fig.savefig(v_io, format='png')
	plt.clf()
	data = base64.encodebytes(v_io.getvalue()).decode()

	return data

def plot_distr(data, cur_len, title = 'Mean text sentence length', b = 0.75):

	weights = np.ones_like(data) / float(len(data))
	binwidth = b

	fig = plt.figure()  # Plotting
	canvas = FigureCanvas(fig)

	plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth), weights=weights)
	plt.axvline(cur_len, color='r', linestyle='dashed', linewidth=3)
	plt.title(title)
	plt.xlabel("Value")
	plt.ylabel("Frequency")

	v_io = io.BytesIO()
	fig.savefig(v_io, format='png')
	plt.clf()
	data = base64.encodebytes(v_io.getvalue()).decode()
	
	return """<img src="data:image/png;base64,%s"/>""" % data

def input_toggle(headline, textik):
	return '<h4 class="clickable-heading">' + headline + '</h4>' + '<ul style="display:none">' + textik + '</ul>'


###### the app itself ###### 
 
app = Flask(__name__)

# https://stackoverflow.com/a/5872904  
class RegexConverter(BaseConverter):
	def __init__(self, url_map, *items):
		super(RegexConverter, self).__init__(url_map)
		self.regex = items[0]
   
app.url_map.converters['regex'] = RegexConverter

from ufal.udpipe import Model, Pipeline, ProcessingError # pylint: disable=no-name-in-module

'''
java_path = "C:/Program Files (x86)/Java/jre1.8.0_161/bin/java.exe"
##################################### HERE
os.environ['JAVAHOME'] = java_path
'''

def get_sentences(intext):	
	sys.stderr.write('Loading model...')
	m = Model.load("english-ud-2.0-170801.udpipe")
	if not m:
		sys.stderr.write("Cannot load model from file english-ud-2.0-170801.udpipe")
		sys.exit(1)
	sys.stderr.write('Done.')

	pipeline = Pipeline(m, "generic_tokenizer", Pipeline.DEFAULT, Pipeline.DEFAULT, "horizontal")
	error = ProcessingError()

	# Process data
	processed = pipeline.process(intext, error)
	if error.occurred():
		sys.stderr.write("An error occurred when running run_udpipe: ")
		sys.stderr.write(error.message)
		sys.stderr.write("\n")
		sys.exit(1)
	return processed

def inspect_verbs(intext):
	sentences = get_sentences(intext)
	tagged = st.tag_sents(nltk.word_tokenize(sent) for sent in sentences.split('\n'))
	out = []
	infinitives = 0
	gerunds = 0
	participles = 0
	total = 0
	wasTo = False
	non_ing = ("alarming", "aggravating", "amusing", "annoying", "astonishing", "astounding", "boring", "captivating", "challenging", "charming", "comforting", "confusing", "convincing", "depressing", "disappointing", "discouraging", "disgusting", "distressing", "disturbing", "embarrassing", "encouraging", "entertaining", "exciting", "exhausting", "fascinating", "frightening", "frustrating", "fulfilling", "gratifying", "inspiring", "insulting", "interesting", "moving", "overwhelming", "perplexing", "pleasing", "relaxing", "relieving", "satisfying", "shocking", "sickening", "soothing", "surprising", "tempting", "terrifying", "threatening", "thrilling", "tiring", "touching", "troubling", "unsettling", "worrying")
	Yellowish = '#ffd900'
	Reddish = '#ff3333'
	Blueish = '#0077ff'
	for a in tagged:
		for l in range(len(a)):
			if a[l][1] == 'VB':
				if l>0 and a[l-1][0].lower() == 'to':
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>1 and a[l-1][1][:2] == 'RB' and a[l-2][0].lower() == 'to':
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>2 and a[l-1][0].lower() == 'least' and a[l-2][0].lower() == 'at' and a[l-3][0].lower() == 'to':
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>3 and a[l-1][0].lower() == 'sometimes' and a[l-2][0].lower() == 'least' and a[l-3][0].lower() == 'at' and a[l-4][0].lower() == 'to':
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>4 and a[l-1][0].lower() == 'while' and a[l-2][0].lower() == 'a' and a[l-3][0].lower() == 'in' and a[l-4][0].lower() == 'once' and a[l-5][0].lower() == 'to':
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>2 and a[l-1][0].lower() == 'day' and a[l-2][0].lower() == 'every' and a[l-3][0].lower() == 'to':
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>2 and a[l-1][0].lower() == 'time' and a[l-2][0].lower() == 'every' and a[l-3][0].lower() == 'to':
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>3 and a[l-1][0].lower() == 'time' and a[l-2][0].lower() == 'the' and a[l-3][0].lower() == 'all' and a[l-4][0].lower() == 'to':
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>3 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'least' and a[l-3][0].lower() == 'at' and a[l-4][0].lower() == 'to':
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>4 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'sometimes' and a[l-3][0].lower() == 'least' and a[l-4][0].lower() == 'at' and a[l-5][0].lower() == 'to':
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>5 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'while' and a[l-3][0].lower() == 'a' and a[l-4][0].lower() == 'in' and a[l-5][0].lower() == 'once' and a[l-6][0].lower() == 'to':
					out[l-6] = (out[l-6][0],Yellowish)
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>3 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'day' and a[l-3][0].lower() == 'every' and a[l-4][0].lower() == 'to':
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>3 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'time' and a[l-3][0].lower() == 'every' and a[l-4][0].lower() == 'to':
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>4 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'time' and a[l-3][0].lower() == 'the' and a[l-4][0].lower() == 'all' and a[l-5][0].lower() == 'to':
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>3 and a[l-1][0].lower() == 'least' and a[l-2][0].lower() == 'at' and a[l-4][0].lower() == 'to' and a[l-3][0].lower()==',':
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>4 and a[l-1][0].lower() == 'sometimes' and a[l-2][0].lower() == 'least' and a[l-3][0].lower() == 'at' and a[l-5][0].lower() == 'to' and a[l-4][0].lower()==',':
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>5 and a[l-1][0].lower() == 'while' and a[l-2][0].lower() == 'a' and a[l-3][0].lower() == 'in' and a[l-4][0].lower() == 'once' and a[l-6][0].lower() == 'to' and a[l-5][0].lower()==',':
					out[l-6] = (out[l-6][0],Yellowish)
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>3 and a[l-1][0].lower() == 'day' and a[l-2][0].lower() == 'every' and a[l-4][0].lower() == 'to' and a[l-3][0].lower()==',':
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>3 and a[l-1][0].lower() == 'time' and a[l-2][0].lower() == 'every' and a[l-4][0].lower() == 'to' and a[l-3][0].lower()==',':
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>4 and a[l-1][0].lower() == 'time' and a[l-2][0].lower() == 'the' and a[l-3][0].lower() == 'all' and a[l-5][0].lower() == 'to' and a[l-4][0].lower()==',':
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>4 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'least' and a[l-3][0].lower() == 'at' and a[l-5][0].lower() == 'to' and a[l-4][0].lower()==',':
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>5 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'sometimes' and a[l-3][0].lower() == 'least' and a[l-4][0].lower() == 'at' and a[l-6][0].lower() == 'to' and a[l-5][0].lower()==',':
					out[l-6] = (out[l-6][0],Yellowish)
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>6 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'while' and a[l-3][0].lower() == 'a' and a[l-4][0].lower() == 'in' and a[l-5][0].lower() == 'once' and a[l-7][0].lower() == 'to' and a[l-6][0].lower()==',':
					out[l-7] = (out[l-7][0],Yellowish)
					out[l-6] = (out[l-6][0],Yellowish)
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>4 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'day' and a[l-3][0].lower() == 'every' and a[l-5][0].lower() == 'to' and a[l-4][0].lower()==',':
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>4 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'time' and a[l-3][0].lower() == 'every' and a[l-5][0].lower() == 'to' and a[l-4][0].lower()==',':
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				elif l>5 and a[l-1][0].lower()==',' and a[l-2][0].lower() == 'time' and a[l-3][0].lower() == 'the' and a[l-4][0].lower() == 'all' and a[l-6][0].lower() == 'to' and a[l-5][0].lower()==',':
					out[l-6] = (out[l-6][0],Yellowish)
					out[l-5] = (out[l-5][0],Yellowish)
					out[l-4] = (out[l-4][0],Yellowish)
					out[l-3] = (out[l-3][0],Yellowish)
					out[l-2] = (out[l-2][0],Yellowish)
					out[l-1] = (out[l-1][0],Yellowish)
					out.append((a[l][0],Yellowish))
					infinitives += 1
				else:
					out.append((a[l][0],None))
			elif a[l][1] == 'VBG' and a[l][0].lower() not in non_ing:
				out.append((a[l][0],Reddish))
				gerunds += 1
			elif a[l][1] == 'VBN' and l!=len(a)-1 and (a[l][0].lower() != 'been' or a[l+1][1][0] != 'V'):
				out.append((a[l][0],Blueish))
				participles += 1
			else:
				out.append((a[l][0],None))
			if a[l][0] != '<br>' and re.search(r'[a-zA-Z0-9]',a[l][0]):
				total += 1
	totalVerbs = gerunds+infinitives+participles
	percGerunds = gerunds/total
	percInfinitives = infinitives/total
	percParticiples = participles/total
	percVerbs = percGerunds+percInfinitives+percParticiples
	return out, percVerbs, totalVerbs, percInfinitives, percParticiples, percGerunds, infinitives, participles, gerunds, tagged
	


###################todo-start######################

@app.errorhandler(404)
def mistake404(code):
	return 'Sorry, this page does not exist!'

#############################todo-end##############################



@app.route('/',methods=['GET','POST']) # or @route('/login', method='POST')
def HelloPage():
	if request.method == 'POST':
		text_type = request.form['type']
		text_to_inspect = request.form['text_to_inspect']
		try:
			stats = TextStats(text_type, text_to_inspect)
			return str(stats)
		except Exception as e:
	#		return "Looks like something wrong happened with your text decoding. Make sure there are regular\
	#					latin characters and regular punctuation symbols in the text."
			sleep(1)
			ErrorStr = """<!doctype html>
			<html lang = "en">
			<script src="/static/engine.js"></script>
		<head>
			<meta charset="utf-8"/>
			<link rel="stylesheet" href="/static/layout.css">
			<title>REALEC Inspector - Error!</title>

		</head>
		<body onload="fadein(document.body,500)">
			<div id="h1">REALEC Inspector</div>
			<div class="txt" style="width: 80%; margin: auto; text-align: center; font-size: 13pt; color: #CC0000;">Either we encountered a decoding error or yet unknown development problem. Tech info: <b>"""
			ErrorStr += str(e)
			ErrorStr += '<br>'
			type_, value_, traceback_ = sys.exc_info()
			ErrorStr += str(traceback.format_tb(traceback_))
			ErrorStr += """</b></div>
			</body>
			</html>"""
			return ErrorStr
	return render_template('index.html')

#[A-Za-z0-9//]+
@app.route('/text-<regex("[^\>]*"):text_to_inspect>')
def inspect_from_address(text_to_inspect):
	# return os.path.dirname(__file__) + str(text_to_inspect)
	path_to_text = os.path.abspath(os.path.dirname(__file__)+'/../realec/') + text_to_inspect
	try:
		with open(path_to_text, 'r', encoding='utf-8-sig') as f:
			text_to_inspect = f.read()
	except Exception as e:
		return "Looks like there is no such file on the server."
		#return 
	try:
				stats = TextStats(text_to_inspect)
				return str(stats)
	except:
		return "Looks like something wrong happened with your text decoding. Make sure there are regular\
					latin characters and regular punctuation symbols in the text."
 
 
#inspect collection
@app.route('/coll-<regex("[^\>]*"):coll_to_inspect>')
def inspect_collection(coll_to_inspect):
	colldir = os.path.abspath(os.path.dirname(__file__)) + coll_to_inspect
	texts = [os.path.join(dp, f) for dp, dn, filenames in os.walk(colldir) for f in filenames if os.path.splitext(f)[1] == '.txt']
	textstats = [TextStats(open(f,'r',encoding='utf-8-sig').read()) for f in texts]
	collstats = CollectionStats(*textstats)
	return str(collstats)