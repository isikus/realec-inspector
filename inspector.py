from flask import Flask, request, render_template
from werkzeug.routing import BaseConverter
import os
#os.environ['MPLCONFIGDIR'] = "/var/www/realec/.matplotlib"
import matplotlib
matplotlib.use('Agg')
import numpy, matplotlib, matplotlib.pyplot as plt
import io
import base64
import sys
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

nltk.data.path.append('/var/www/inspector/nltk_data/')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag, pos_tag_sents

from nltk import StanfordPOSTagger

wordnet_lemmatizer = WordNetLemmatizer()
st = StanfordPOSTagger('english-bidirectional-distsim.tagger','stanford-postagger.jar')

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
        def __init__(self,text_to_inspect):

                self.text = text_to_inspect
                
                self.verbshtml, self.percVerbs, self.totalVerbs, self.percInfinitives, self.percParticiples, \
                self.percGerunds, self.infinitives, self.participles, self.gerunds = inspect_verbs(self.text)

                self.text = re.sub(r'[^\x00-\x7F]+','', self.text)

                self.clean_words_list, self.stats, self.coca_freq, self.academic, self.num_of_words, self.levels, \
                self.overall_counter, self.intro_group_counter, self.intro_group_dict, self.intro_phrase_counter, \
                self.intro_groups, self.tokens, self.collocations, self.stopwords_in_text = GetStats(self.text)
                
                self.ff = DealWithFreq(self.coca_freq)
                self.num_of_repetitions = sum(self.overall_counter.values()) - len(self.overall_counter)
                self.most_common_repetition  = self.overall_counter.most_common(1)[0]

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
        
        

        def __str__(self):
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
                        ShorterError += """</b> words. The current IELTS standard requires that an essay in Task 1 should have not less than 150 words. Please go back to your essay and complete the text.</div>
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
                        <div class="txt" style="width: 80%; margin: auto; text-align: center; font-size: 13pt; font-weight: bold;">Here is your text:</div>
                        <div class="txt" id="hiddenplacer" style="position: relative; width: 98%; margin: auto; text-align: center; z-index: 0; visibility: hidden;">"""
        #	ResultPage += """<h2>Source Essay</h2><br>"""
        #	with open ("/home/ivan/outt.txt","w",encoding="utf-8") as t:
        #		t.write(text_to_inspect)
                
                ResultPage += re.sub(r'( )*<br>( )*',r'<br>',html_text)

                ResultPage += """<div class="txt" id="textplacer" style="position: absolute; width: 100%; top: 0px; left: 0px; text-align: center; visibility: visible; z-index: 0;">"""
                ResultPage += re.sub(r'( )*<br>( )*',r'<br>',html_text)

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

                Red = "#CC0000"
                Green = "#00CC00"
                Blue = "#0000CC"

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
                                words0 += textcolor(splitted_text[i],Blue,self.tokens[k].is_freq(0)) + ' '

                ResultPage += """<div class="txt" id="words_500" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
                ResultPage += re.sub(r'( )*<br>( )*',r'<br>',words500)
                ResultPage += """</div>
                        <div class="txt" id="words_3000" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
                ResultPage += re.sub(r'( )*<br>( )*',r'<br>',words3000)
                ResultPage += """</div>
                        <div class="txt" id="words_0" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
                ResultPage += re.sub(r'( )*<br>( )*',r'<br>',words0)


        #	ResultPage += ('<br><br><h3>Academic Words</h3>')

                academic_words = str()

                k=-1
                for i in range(len(splitted_text)):
                        k=k+1
                        '''
                        try:
                                        ord(splitted_text[i])
                        except:
                                try:
                                        academic_words += '&#'+str(ord(splitted_text[i].decode('utf-8'))) + ' '
                                        k=k-1
                                        continue
                                except:
                                        pass
                        '''
                        if not any(ord(char) < 128 for char in splitted_text[i]):
                                academic_words += splitted_text[i] + ' '
                                k=k-1
        #			continue
                        elif splitted_text[i] == '<br>':
                                k=k-1
                                academic_words = academic_words[:-1]+splitted_text[i]
        #			continue
                        else:
                                academic_words += textcolor(splitted_text[i],Green,self.tokens[k].is_academic()) + ' '

                ResultPage += """</div>
                        <div class="txt" id="academic" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
                ResultPage += re.sub(r'( )*<br>( )*',r'<br>',academic_words)
                ResultPage += """</div>
                        <div class="txt" id="spelling" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
                ### SPELLING GOES HERE
                ResultPage += """</div>
                        <div class="txt" id="verbs" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
                ResultPage += re.sub(r'( )*<br>( )*',r'<br>',self.verbshtml)
                ResultPage += """</div>
                        <div class="txt" id="linking" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
                
                linking_text = html_text
                for elem in self.intro_groups.keys():
                        if self.intro_groups[elem] != 0:
                                for phrase in set(self.intro_group_dict[elem]):
                                        replstr = re.compile('('+phrase+')', re.IGNORECASE)
                                        linking_text = replstr.sub(r'<font color="'+Green+r'">\1</font>',linking_text)
                ResultPage += re.sub(r'( )*<br>( )*',r'<br>',linking_text)
                
                ResultPage += """</div>
                        <div class="txt" id="collocations" name="textmask" style="position: absolute; width: 100%; top: 0px; left: 0px; font-size: 11pt; text-align: center; color: #222222; visibility: hidden;">"""
                
                colloc_text = html_text
                for colloc in self.collocations.keys():
                        replstr = re.compile('('+colloc+')', re.IGNORECASE)
                        colloc_text = replstr.sub(r'<font color="'+Green+r'">\1</font>',colloc_text)
                ResultPage += re.sub(r'( )*<br>( )*',r'<br>',colloc_text)
                        
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
                
                ### SPELLING
                ### SPELLING IS OMITTED

                '''
                ResultPage += """			<tr id="spelling_tr" """
                if spelling=="good":
                        ResultPage += """class="red">
                                        <td valign="top" style="width: 12%">
                                                Spelling
                                        </td>
                                        <td style="width: 100%">
                                                Unfortunately a <a class="innerlink" id="spellinglink" onclick="reversemaskerlink(this,'spelling')" href="javascript:showmasker('spelling')">significant portion</a> of your words cannot be identified. Try to correct spelling errors and possibly use more conventional vocabulary.
                                        </td>
                                </tr>"""
                else:
                        ResultPage += """class="green">
                                        <td valign="top" style="width: 12%">
                                                Spelling
                                        </td>
                                        <td style="width: 100%">
                                        # You should revise that!
                                                Your spelling seems rather good. Keep up the good work!
                                        </td>
                                </tr>"""
                '''
                
                ### ACADEMIC
                ResultPage += """			<tr id="academic_tr" """
                
                if self.num_academic/(len(self.tokens)-len(self.stopwords_in_text)) > 0.33333:
                        ResultPage += """class="green">
                                        <td valign="top" style="width: 12%">
                                                Academic words
                                        </td>
                                        <td style="width: 100%">
                                                The use of <a class="innerlink" id="academiclink" onclick="reversemaskerlink(this,'academic')" href="javascript:showmasker('academic')">academic vocabulary</a> in your essay is impressive.
                                        </td>
                                </tr>"""

                else:
                        ResultPage += """class="red">
                                        <td valign="top" style="width: 12%">
                                                Academic words
                                        </td>
                                        <td style="width: 100%">
                                                Highly rated essays tend to have more <a class="innerlink" id="academiclink" onclick="reversemaskerlink(this,'academic')" href="javascript:showmasker('academic')">academic words</a> than yours. See the list of academic vocabulary <a href="https://www.vocabulary.com/lists/1220858">here</a>.
                                        </td>
                                </tr>"""


                ### VERBS
                
                ResultPage += """			<tr id="verbs_tr" """
                if self.percVerbs<0.0344:
                        ResultPage += """class="red">
                                        <td valign="top" style="width: 12%">
                                                Verb usage
                                        </td>
                                        <td style="width: 100%">
                                                Highly rated essays have a bigger proportion of <a class="innerlink" id="verbslink" onclick="reversemaskerlink(this,'verbs')" href="javascript:showmasker('verbs')"><div class="yellowish" style="display: inline-block">infinitival</div>, <div class="reddish" style="display: inline-block">gerundival</div> and <div class="blueish" style="display: inline-block">participial</div></a> constructions. Keep the need to apply them in mind when you write your next essay.
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

                ### LINKING
                ResultPage += """			<tr id="linking_tr" """

                if sum(self.intro_groups.values()) >= 4:
                        ResultPage += """class="green">
                                        <td valign="top" style="width: 12%">
                                                Linking phrases
                                        </td>
                                        <td style="width: 100%">
                                                A fair number of <a class="innerlink" id="linkinglink" onclick="reversemaskerlink(this,'linking')" href="javascript:showmasker('linking')">linking tools</a> have been identified in your essay. Well done! Make sure you apply them appropriately.
                                        </td>
                                </tr>"""
                else:
                        ResultPage += """class="red">
                                                <td valign="top" style="width: 12%">
                                                        Linking phrases
                                                </td>
                                                <td style="width: 100%">
                                                        Think where you could use more <a class="innerlink" id="linkinglink" onclick="reversemaskerlink(this,'linking')" href="javascript:showmasker('linking')">linking devices</a>.
                                                </td>
                                        </tr>"""	

                ### COLLOCATIONS
                ResultPage += """			<tr id="collocations_tr" """	

                if sum(self.collocations.values()) >= 2:
                        ResultPage += """class="green">
                                        <td valign="top" style="width: 12%">
                                                Collocations
                                        </td>
                                        <td style="width: 100%">
                                                You did a good job with <a class="innerlink" id="collocationslink" onclick="reversemaskerlink(this,'collocations')" href="javascript:showmasker('collocations')">collocations</a>, which guarantees a high level of a part of the grade.
                                        </td>
                                </tr>"""
                else:
                        ResultPage += """class="red">
                                                <td valign="top" style="width: 12%">
                                                        Collocations
                                                </td>
                                                <td style="width: 100%">
                                                        You need to use more <a class="innerlink" id="collocationslink" onclick="reversemaskerlink(this,'collocations')" href="javascript:showmasker('collocations')">collocations</a>.
                                                </td>
                                        </tr>"""	
                

                ### SYNTAX
                ### SYNTAX IS OMITTED

                '''
                ResultPage += """			<tr id="spelling_tr" """
                if syntax=="good":
                        ResultPage += """class="red">
                                        <td valign="top" style="width: 12%">
                                                Syntax
                                        </td>
                                        <td style="width: 100%">
                                                Good syntax!
                                        </td>
                                </tr>"""
                else:
                        ResultPage += """class="green">
                                        <td valign="top" style="width: 12%">
                                                Spelling
                                        </td>
                                        <td style="width: 100%">
                                                Bad syntax:(
                                        </td>
                                </tr>"""
                '''
                
                
                ### LENGTH
                ResultPage += """			<tr id="length_tr" """
                
                if float(getPerc(mean_sent_lengths, self.MeanSentLength)) > 60:
                        ResultPage += """class="green">
                                        <td valign="top" style="width: 12%">
                                                Sentence length
                                        </td>
                                        <td style="width: 100%">
                                                You have <a class="innerlink" id="lengthlink" onclick="popuplength(event)" href="javascript:void(0)">sufficiently</a> complex sentences in your essay. Keep it up!
                                        </td>
                                </tr>"""
                else:
                        ResultPage += """class="red">
                                                <td valign="top" style="width: 12%">
                                                        Sentence length
                                                </td>
                                                <td style="width: 100%">
                                                        The mean sentence length in the best essays is higher than <a class="innerlink" id="lengthlink" onclick="popuplength(event)" href="javascript:void(0)">in yours</a>. Try using more complex sentences.
                                                </td>
                                        </tr>"""	



        #	if float(getPerc(mean_word_lengths, GetMeanWordLength(text_to_inspect))) < 60 and IsCEFRRec(stats):
        #		ResultPage += ('The words are mostly of a basic level. Use more sophisticated vocabulary.\
        #					 Also, more academic words are needed.\
        #					See <a href="https://www.vocabulary.com/lists/1220858">academic vocabulary 100.</a>')
        #	elif float(getPerc(mean_word_lengths, GetMeanWordLength(text_to_inspect))) > 60 and not IsCEFRRec(stats):
        #		ResultPage += ('The vocabulary in your essay is impressive.')
        #	else:
        #		ResultPage += ('Also, more academic words are needed.\
        #					See <a href="https://www.vocabulary.com/lists/1220858">academic vocabulary 100.</a> ')
        #	ResultPage += 'There are too many unclassified words which must be checked for spelling errors. '

                ################# END OF EVALUATION PART ##################


                ResultPage += """		</table>
                        <div class="txt" id="showcut" style="width: 80%; margin: auto; text-align: center; font-size: 9pt;  display: block;"><a href="javascript:showcut()">click to show advanced statistical data &#8595;</a></div>
                        <div class="txt" id="hidecut" style="width: 80%; margin: auto; text-align: center; font-size: 9pt; opacity: 0; display: none;"><a href="javascript:hidecut()">hide advanced data &#8593;</a></div>"""


                ################# NEATY LITTLE SENTENCE LENGTH ##################

                ResultPage += """<div class="txt" id="length" style="position: absolute; width: auto; top: 0px; left: 0px; font-size: 10px; background-color: white; border: 3px dashed; border-color: rgb(190,16,123); border-radius: 5px; opacity: 0; display: none;">"""
                ResultPage += str(round(self.MeanSentLength,2))
                ResultPage += """ words</div>"""

                ################# LENGTH END ###################







                ################# BEGINNING OF CUT PART ###################


                ResultPage += """
                        <div class="txt" id="cut" style="display: block; font-size: 11pt; display: none;">"""


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
			intro_groups[group] += 1
		elif word + ' ' + next_word in introwords:
			phrase = word + ' ' + next_word
			intro_phrase_counter[phrase] += 1
			group = introwords_hash[phrase]
			intro_group_counter[group] += 1
			intro_group_dict[group].append(phrase)
			intro_groups[group] += 1
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
				intro_group_counter, intro_group_dict, intro_phrase_counter, intro_groups, tokens, collocations, stopwords_in_text
	else:
		return clean_words_list, eng_voc_stats, coca_freq, academic, num_of_words, levels_ev, overall_counter,\
				intro_group_counter, intro_group_dict, intro_phrase_counter, intro_groups, tokens, collocations, stopwords_in_text

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

def inspect_verbs(intext):
	bufftxt = re.sub(r'\n',' <br> ',intext)
	buffarr = bufftxt.split()
	now = time()
	tagged = st.tag(intext.split())
	print('tagging',time()-now)
	out = ""
	infinitives = 0
	gerunds = 0
	participles = 0
	total = 0
	wasTo = False
	b = -1
	for l in range(len(tagged)):
		b = b + 1
		if buffarr[b] == '<br>':
			out += "<br>"
			b = b + 1
		if tagged[l][1] == 'TO':
			wasTo = True
			total += 1
			continue
		if wasTo:
			if tagged[l][1] == 'VB':
				out += textcolor(tagged[l-1][0],'#ffd900',True)
				out += ' '
				out += textcolor(tagged[l][0],'#ffd900',True)
				out += ' '
				infinitives += 1
				wasTo = False
				total += 1
				continue
			else:
				out += tagged[l-1][0]
				out += ' '
				wasTo = False
		if tagged[l][1] == 'VBG':
			out += textcolor(tagged[l][0],'#ff3333',True)
			gerunds += 1
		elif tagged[l][1] == 'VBN' and (tagged[l][0] != 'been' or tagged[l+1][1][0] != 'V'):
			out += textcolor(tagged[l][0],'#0077ff',True)
			participles += 1
		else:
			out += tagged[l][0]
		out += ' '
		if tagged[l][0] != '<br>':
			total += 1
	totalVerbs = gerunds+infinitives+participles
	percGerunds = gerunds/total
	percInfinitives = infinitives/total
	percParticiples = participles/total
	percVerbs = percGerunds+percInfinitives+percParticiples

	return out, percVerbs, totalVerbs, percInfinitives, percParticiples, percGerunds, infinitives, participles, gerunds
    
    
###################todo-start######################

@app.errorhandler(404)
def mistake404(code):
    return 'Sorry, this page does not exist!'

###################################todo-end########################



@app.route('/',methods=['GET','POST']) # or @route('/login', method='POST')
def HelloPage():
    if request.method == 'POST':
        text_to_inspect = request.form['text_to_inspect']
        try:
            now = time()
            stats = TextStats(text_to_inspect)
            print('total',time()-now)
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

if __name__ == '__main__':
	app.run(debug=True)