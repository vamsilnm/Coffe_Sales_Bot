import json
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from duckling import DucklingWrapper
from datetime import datetime
from datetime import timedelta,date
from dateutil import parser
from nltk.stem.wordnet import WordNetLemmatizer 
import sys
import re
from nltk.classify import Senna
from nltk.stem.wordnet import WordNetLemmatizer
pipeline = Senna('/usr/share/senna-v3.0', ['pos', 'chk', 'ner'])
lmtzr = WordNetLemmatizer()

#Stop Words generation
stop_words = []
file_open_stopwords = open('stop_words.txt','r')
for each_word in file_open_stopwords:
	stop_words.append(each_word.strip())
file_open_stopwords.close()

d = DucklingWrapper()
leave_application_json = {}
context_stack = []
leave_application_previous = {}
def intent_extractor(message):
	file_Name = "intent_leave_tfidf.p"
	fileObject = open(file_Name,'rb') 
	passive_tfidf = pickle.load(fileObject)
	fileObject.close()

	file_Name_lsa = "intent_leave_lsa.p"
	fileObject_lsa = open(file_Name_lsa,'rb') 
	passive_lsa = pickle.load(fileObject_lsa)
	fileObject_lsa.close()

	file_vectorizer_open = open("tfidf_vectorizer.p",'rb')
	vectorizer = pickle.load(file_vectorizer_open)
	file_vectorizer_open.close()

	file_lsa_vectorizer_open = open("lsa_vectorizer.p",'rb')
	lsa = pickle.load(file_lsa_vectorizer_open)
	file_lsa_vectorizer_open.close()

	out_put = []
	test_text = message
	test_text_clean = [each_word for each_word in test_text.split() if each_word not in stop_words] 
	test_text_lmtzr = [lmtzr.lemmatize(each_word) for each_word in test_text_clean]
	out_put.append(' '.join(test_text_lmtzr))
	out_put_vector = vectorizer.transform(out_put)
	out_put_class = passive_tfidf.predict(out_put_vector)

	out_put_vector_lsa = lsa.transform(out_put_vector)


	tfidf_class = None
	lsa_class = None
	tfidf_confidence =  list(passive_tfidf.decision_function(out_put_vector)[0])
	lsa_confidence = list(passive_lsa.decision_function(out_put_vector_lsa)[0])

	for each_item in tfidf_confidence:
		if each_item > 0:
			# print each_item
			tfidf_class = out_put_class[0]
	for each_item in lsa_confidence:
		if each_item > 0:
			# print each_item
			lsa_class = passive_lsa.predict(out_put_vector_lsa)[0]


	return lsa_class


def get_global_intent(message):
	file_Name = "global_intent_tfidf.p"
	fileObject = open(file_Name,'rb') 
	passive_tfidf = pickle.load(fileObject)
	fileObject.close()

	file_Name_lsa = "global_intent_lsa.p"
	fileObject_lsa = open(file_Name_lsa,'rb') 
	passive_lsa = pickle.load(fileObject_lsa)
	fileObject_lsa.close()

	file_vectorizer_open = open("global_intent_tfidf_vectorizer.p",'rb')
	vectorizer = pickle.load(file_vectorizer_open)
	file_vectorizer_open.close()

	file_lsa_vectorizer_open = open("global_intent_lsa_vectorizer.p",'rb')
	lsa = pickle.load(file_lsa_vectorizer_open)
	file_lsa_vectorizer_open.close()

	out_put = []
	test_text = message
	test_text_clean = [each_word for each_word in test_text.split() if each_word not in stop_words] 
	test_text_lmtzr = [lmtzr.lemmatize(each_word) for each_word in test_text_clean]
	out_put.append(' '.join(test_text_lmtzr))
	out_put_vector = vectorizer.transform(out_put)
	out_put_class = passive_tfidf.predict(out_put_vector)
	out_put_vector_lsa = lsa.transform(out_put_vector)
	tfidf_class = None
	lsa_class = None
	tfidf_confidence =  list(passive_tfidf.decision_function(out_put_vector)[0])
	lsa_confidence = list(passive_lsa.decision_function(out_put_vector_lsa)[0])

	for each_item in tfidf_confidence:
		if each_item > 0:
			# print each_item
			tfidf_class = out_put_class[0]
	for each_item in lsa_confidence:
		if each_item > 0:
			# print each_item
			lsa_class = passive_lsa.predict(out_put_vector_lsa)[0]


	return lsa_class


def sentence_tokens_generation(message):
	sentence_tokens = []
	sentence = message
	word_tokens = sentence.split()
	word_tokens_clean = [each_word.strip() for each_word in word_tokens]
	word_tokens_pos_tagged_containing_stopwords = [(token['word'], token['chk'], token['ner'], token['pos']) for token in pipeline.tag(word_tokens_clean)]
	word_tokens_pos_tagged = [word for word in word_tokens_pos_tagged_containing_stopwords if word[0] not in stop_words]	
	each_sentence_tokens = []
	for each_token in word_tokens_pos_tagged:
		each_sentence_tokens.append(each_token)
	sentence_tokens.append(each_sentence_tokens)

	return sentence_tokens

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2tokens(sent):
    return [token for token, chunk, ner_tag, postag, label in sent]

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][3]
    chunk = sent[i][1]
    ner_tag = sent[i][2]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'chk':chunk,
        'ner_tag':ner_tag,        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features

def entity_extractor(message):
	file_open_pickle = open('entity_model_leave.p','rb')
	crf = pickle.load(file_open_pickle) 
	
	message_tokens = sentence_tokens_generation(message)
	message_vector = [sent2features(s) for s in message_tokens]
	message_entity = crf.predict(message_vector)

	entity_list = []
	for each_sentence in zip(message_tokens,message_entity):
		for each_word in zip(each_sentence[0],each_sentence[1]):
			entity_list.append((each_word[0][0],each_word[1]))

	return entity_list

def is_duration(entity_list):
	duration_number = ''
	duration_type = ''
	for each_entity in entity_list:
		if each_entity[1] == 'du':
			duration_number = each_entity[0]
		if not duration_type:
			if lmtzr.lemmatize(each_entity[0].strip().lower()) == 'day':
				duration_type = 'day'
			elif lmtzr.lemmatize(each_entity[0].strip().lower()) == 'month':
				duration_type = 'month'
			elif lmtzr.lemmatize(each_entity[0].strip().lower()) == 'year':
				duration_type = 'year'
			elif lmtzr.lemmatize(each_entity[0].strip().lower()) == 'week':
				duration_type = 'week'

	if duration_type in ('week','year','month','day') and duration_number:
		return 'for '+ duration_number + ' ' +duration_type
	elif  duration_type in ('week','year','month','day'):
		return 'for ' + 'a ' + duration_type
	return None


def is_date(entity_list):
	date = ''
	date_presence = 0
	for each_entity in entity_list:
		if each_entity[1] == 'dt':
			date_presence = 1
			date += ' ' +each_entity[0]
		if each_entity[1] == 'dt_i':
			date += ' '+ each_entity[0]
	return date

def parsing_date(message_parsed):
	if not message_parsed['value'].get('grain'):
		if message_parsed['value']['value'].get('to'):
			return parser.parse(message_parsed['value']['value'].get('to'))
		elif message_parsed['value']['value'].get('from'):
			return parser.parse(message_parsed['value']['value'].get('from'))
	elif message_parsed['value'].get('grain'):
		return parser.parse(message_parsed['value']['value'])
	else:
		return None

def date_extractor(message):
	global context_stack
	leave_dates = {}
	try:
		probable_dates = []
		for each_date in re.finditer(r'((\d+/\d+/\d+)|(\d+-\d+-\d+))', message):
			try:
				try:
					probable_dates.append(datetime.strptime(each_date.group(1), "%d/%m/%Y"))
				except:
					probable_dates.append(datetime.strptime(each_date.group(1), "%d-%m-%Y"))
			except:
				print 'Please restrict your self to date month and year format only.'
				return
		if len(probable_dates) == 2:
			leave_dates['start_date'] = probable_dates[0]
			leave_dates['end_date'] = probable_dates[1] + timedelta(days=1)
			return leave_dates
		elif len(probable_dates) == 1:
			message_intent = intent_extractor(message)
			message_entities = entity_extractor(message)
			if message_intent == 'leave_start_date_present':
				if is_duration(message_entities):
					days_to_be_added = parsing_date(d.parse_time(is_duration(message_entities))[0]).date() - datetime.today().date()
					leave_dates['start_date'] = probable_dates[0]
					leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days=days_to_be_added.days+1)
					return leave_dates
				else:
					leave_dates['start_date'] = probable_dates[0]
					return leave_dates
			elif message_intent == 'leave_end_date_present':
				if is_duration(message_entities):
					days_to_be_subtracted = parsing_date(d.parse_time(is_duration(message_entities))[0]).date() - datetime.today().date()
					leave_dates['end_date'] = probable_dates[0] + timedelta(days=1)
					leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days=-1*days_to_be_subtracted.days)
					return leave_dates
				else:
					leave_dates['end_date'] = probable_dates[0]+ timedelta(days=1)
					return leave_dates
			else:
				try:
					if context_stack[-1] == 'start_date':
						leave_dates['start_date'] = probable_dates[0]
						return leave_dates
					elif context_stack[-1] == 'end_date':
						leave_dates['end_date'] = probable_dates[0] + timedelta(days=1)
						return leave_dates
					else:
						leave_dates['start_date'] = probable_dates[0]
						leave_dates['end_date'] = probable_dates[0] + timedelta(days=1)
						return leave_dates
				except:
					leave_dates['start_date'] = probable_dates[0]
					leave_dates['end_date'] = probable_dates[0] + timedelta(days=1)
					return leave_dates
		else:
			message_parsed = d.parse_time(message)
			if len(message_parsed) == 1:
				if not message_parsed[0]['value'].get('grain'):
					if message_parsed[0]['value']['value'].get('to') and message_parsed[0]['value']['value'].get('from'):
						end_date_string = message_parsed[0]['value']['value'].get('to').split('T')[0]
						leave_dates['end_date'] = end_date_string
						start_date_string = message_parsed[0]['value']['value'].get('from').split('T')[0]
						leave_dates['start_date'] = start_date_string
						if parser.parse(leave_dates['start_date']).date() == parser.parse(leave_dates['end_date']).date():
							leave_dates.clear()
				else:
					message_intent = intent_extractor(message)
					if message_intent != 'leave_start_date_present' and message_intent != 'leave_end_date_present':
						message_entities = entity_extractor(message)
						is_single_day = 1
						for each_entity in message_entities:
							if each_entity[1] == 'du':
								is_single_day = 0
								break
							if lmtzr.lemmatize(each_entity[0].strip().lower()) in ('week','day','year'):
								is_single_day = 0
								break
						if is_single_day:
							if len(context_stack):
								if context_stack[-1] != 'start_date' and context_stack[-1] != 'end_date':
									leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
									leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date() + timedelta(days=1)
									return leave_dates
							else:
								leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
								leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date() + timedelta(days=1)
								return leave_dates

						if context_stack[-1] == 'start_date':
							leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
							return leave_dates
						elif context_stack[-1] == 'end_date':
								leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
								return leave_dates
					if message_intent in ('leave_start_date_present','leave_end_date_present'):
						text_entities = entity_extractor(message_parsed[0]['text'])
						is_text_duration = is_duration(text_entities)
						date = is_date(text_entities)
						message_entities = entity_extractor(message)
						duration_number_message = is_duration(message_entities)
						if is_text_duration and date:
							if message_intent == 'leave_end_date_present':
								try:
									if context_stack[-1] == 'end_date':
										leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
										return leave_dates
									elif context_stack[-1] == 'start_date':
										leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
										return leave_dates
									else:
										leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
										text_parsed = d.parse_time(date)
										if len(text_parsed) == 1:
											leave_dates['end_date'] = parser.parse(text_parsed[0]['value']['value'])
								except:
									leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
									text_parsed = d.parse_time(date)
									if len(text_parsed) == 1:
										leave_dates['end_date'] = parser.parse(text_parsed[0]['value']['value'])
							elif message_intent == 'leave_start_date_present':
								try:
									if context_stack[-1] == 'end_date':
										leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
										return leave_dates
									elif context_stack[-1] == 'start_date':
										leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
										return leave_dates
									else:
										leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
										text_parsed = d.parse_time(date)
										if len(text_parsed) == 1:
											leave_dates['start_date'] = parser.parse(text_parsed[0]['value']['value'])
								except:
									leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
									text_parsed = d.parse_time(date)
									if len(text_parsed) == 1:
										leave_dates['start_date'] = parser.parse(text_parsed[0]['value']['value'])
						elif date and duration_number_message:
							if message_intent == 'leave_start_date_present':
								leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
								duration_number_message_parsed = d.parse_time(duration_number_message)
								days_to_be_added = parser.parse(duration_number_message_parsed[0]['value']['value']).date() - datetime.today().date()
								leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days=days_to_be_added.days)
							elif message_intent == 'leave_end_date_present':
								leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
								duration_number_message_parsed = d.parse_time(duration_number_message)
								days_to_be_subtracted = parser.parse(duration_number_message_parsed[0]['value']['value']).date() - datetime.today().date()
								leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days=-1*days_to_be_subtracted.days)
						else:
							try:
								if context_stack[-1] == 'start_date':
									leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
								elif context_stack[-1] == 'end_date':
									leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date() + timedelta(days=1)
								return leave_dates
							except:
								return None
			elif len(message_parsed) == 2:
				message_entities = entity_extractor(message)
				first_element = message_parsed[0]['text']
				second_element = message_parsed[1]['text']
				first_element_entites = entity_extractor(first_element)
				second_element_entities = entity_extractor(second_element)
				message_intent = intent_extractor(message)
				first_element_date = is_date(first_element_entites)
				second_element_date = is_date(second_element_entities)
				first_element_duration = is_duration(first_element_entites)
				second_element_duration = is_duration(second_element_entities)

				if first_element_date and first_element_duration:
					if message_intent == 'leave_start_date_present':
						leave_dates['end_date'] = parsing_date(message_parsed[0]).date()
						if second_element_duration:
							second_element_duration_parsed = d.parse_time(second_element_duration)
							days_to_be_subtracted = parsing_date(second_element_duration_parsed[0]).date() - datetime.today().date()
							leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days= -1* days_to_be_subtracted.days)
						elif first_element_duration:
							first_element_duration_parsed = d.parse_time(first_element_duration)
							days_to_be_subtracted = parsing_date(first_element_duration[0]).date() - datetime.today().date()
							leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days= -1* days_to_be_subtracted.days)
					elif message_intent == 'leave_end_date_present':
						leave_dates['start_date'] = parsing_date(message_parsed[0]).date()
						if second_element_duration:
							second_element_duration_parsed = d.parse_time(second_element_duration)
							days_to_be_added= parsing_date(second_element_duration_parsed[0]).date() - datetime.today().date()
							leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days= days_to_be_added.days)
						elif first_element_duration:
							first_element_duration_parsed = d.parse_time(first_element_duration)
							days_to_be_added = parsing_date(first_element_duration_parsed[0]).date() - datetime.today().date()
							leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days= days_to_be_added.days)
				elif second_element_duration and second_element_date:
					if message_intent == 'leave_start_date_present':
						leave_dates['end_date'] = parsing_date(message_parsed[1]).date()
						if second_element_duration:
							second_element_duration_parsed = d.parse_time(second_element_duration)
							days_to_be_subtracted = parsing_date(second_element_duration_parsed[0]).date() - datetime.today().date()
							leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days= -1* days_to_be_subtracted.days)
						elif first_element_duration:
							first_element_duration_parsed = d.parse_time(first_element_duration)
							days_to_be_subtracted = parsing_date(first_element_duration_parsed[0]).date() - datetime.today().date()
							leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days= -1* days_to_be_subtracted.days)
					elif message_intent == 'leave_end_date_present':
						leave_dates['start_date'] = parsing_date(message_parsed[1]).date()
						if second_element_duration:
							second_element_duration_parsed = d.parse_time(second_element_duration)
							days_to_be_added= parsing_date(second_element_duration_parsed[0]).date() - datetime.today().date()
							leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days= days_to_be_added.days)
						elif first_element_duration:
							first_element_duration_parsed = d.parse_time(first_element_duration)
							days_to_be_added = parsing_date(first_element_duration_parsed[0]).date() - datetime.today().date()
							leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days= days_to_be_added.days)
				elif first_element_duration and second_element_date:
					if message_intent == 'leave_start_date_present':
						leave_dates['start_date'] = parsing_date(message_parsed[1]).date()
						first_element_duration_parsed = d.parse_time(first_element_duration)
						days_to_be_added = parsing_date(first_element_duration_parsed[0]).date() - datetime.today().date()
						leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days= days_to_be_added.days)
					elif message_intent == 'leave_end_date_present':
						leave_dates['end_date'] = parsing_date(message_parsed[1]).date()
						first_element_duration_parsed = d.parse_time(first_element_duration)
						days_to_be_subtracted = parsing_date(first_element_duration_parsed[0]).date() - datetime.today().date()
						leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days= -1* days_to_be_subtracted.days)
				elif first_element_date and second_element_duration:
					if message_intent == 'leave_end_date_present':
						leave_dates['end_date'] = parsing_date(message_parsed[0]).date()
						second_element_duration_parsed = d.parse_time(second_element_duration)
						days_to_be_subtracted = parsing_date(second_element_duration_parsed[0]).date() - datetime.today().date()
						leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days= -1* days_to_be_subtracted.days)
					elif message_intent == 'leave_start_date_present':
						leave_dates['start_date'] = parsing_date(message_parsed[0]).date()
						second_element_duration_parsed = d.parse_time(second_element_duration)
						days_to_be_added = parsing_date(second_element_duration_parsed[0]).date() - datetime.today().date()
						leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days= days_to_be_added.days)
				elif is_duration(message_entities):
					if message_intent == 'leave_start_date_present':
						if first_element_date:
							leave_dates['start_date'] = parsing_date(message_parsed[0]).date()
						elif second_element_date:
							leave_dates['start_date'] = parsing_date(message_parsed[1]).date()
						if leave_dates.get('start_date'):
							days_to_be_added = parsing_date(d.parse_time(is_duration(message_entities))[0]).date() - datetime.today().date()
							leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days=days_to_be_added.days)
					elif message_intent == 'leave_end_date_present':
						if first_element_date:
							leave_dates['end_date'] = parsing_date(message_parsed[0]).date()
						elif second_element_date:
							leave_dates['end_date'] = parsing_date(message_parsed[1]).date()
						if leave_dates.get('end_date'):
							days_to_be_subtracted = parsing_date(d.parse_time(is_duration(message_entities))[0]).date() - datetime.today().date()
							leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days=-1*days_to_be_subtracted.days)
			return leave_dates
	except Exception,e:
		print e
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print (exc_type,exc_tb.tb_lineno)
		# print message_parsed


def leave_type_extractor(message_entites):
	for each_entity in message_entites:
		if each_entity[1] == 'lt':
			return each_entity[0]
	return None


def leave_application(message):
	global context_stack
	message_entites = entity_extractor(message)
	message_intent = intent_extractor(message)
	if leave_type_extractor(message_entites):
		leave_application_json['leave_type'] = leave_type_extractor(message_entites)
	if date_extractor(message):
		date_extracted = date_extractor(message)
		if date_extracted.get('start_date'):
			leave_application_json['start_date'] = date_extracted['start_date']
		if date_extracted.get('end_date'):
			leave_application_json['end_date'] = date_extracted['end_date']
	if not leave_application_json.get('leave_type'):
		context_stack.append('sick|annual|casual')
		user_response_leave_type = raw_input('Bot: Can you please specify type of leave.\nCasual/Annual/Sick\nUser: ')
		user_response_leave_type_entites = entity_extractor(user_response_leave_type)
		
		if leave_type_extractor(user_response_leave_type_entites):
			leave_application_json['leave_type'] = leave_type_extractor(user_response_leave_type_entites)
		elif user_response_leave_type.strip().lower() in context_stack[-1].split('|'):
			leave_application_json['leave_type'] = user_response_leave_type.strip().lower()
		else:
			print 'Bot: Sorry for not being able to serve you now'
			leave_application_json.clear()
			return
	if not leave_application_json.get('start_date') and not leave_application_json.get('end_date'):
		user_response_date = raw_input('Bot: Can you please help me with start and end dates of leave.\nUser: ')
		date_extracted_second_time = date_extractor(user_response_date)
		if date_extracted_second_time:
			if date_extracted_second_time.get('start_date'):
				leave_application_json['start_date'] = date_extracted_second_time['start_date']
			if date_extracted_second_time.get('end_date'):
				leave_application_json['end_date'] = date_extracted_second_time['end_date']
	if not leave_application_json.get('start_date'):
		user_response_start_date = raw_input('Bot: Can you please specify starting date of leave duration.\nUser: ')
		context_stack.append('start_date')
		start_date_extracted = date_extractor(user_response_start_date)
		try:
			if start_date_extracted.get('start_date'):
				leave_application_json['start_date'] = start_date_extracted.get('start_date')
			if start_date_extracted.get('end_date') and not leave_application_json.get('end_date'):
				leave_application_json['end_date'] = start_date_extracted.get('end_date')
		except:
			user_response_start_date = raw_input('Bot: Can you please specify starting date of leave duration.\nUser: ')
			context_stack.append('start_date')
			start_date_extracted = date_extractor(user_response_start_date)
			try:
				if start_date_extracted.get('start_date'):
					leave_application_json['start_date'] = start_date_extracted.get('start_date')
				if start_date_extracted.get('end_date') and not leave_application_json.get('end_date'):
					leave_application_json['end_date'] = start_date_extracted.get('end_date')
			except:
				print 'Bot: Sorry for not being able to serve you right now.\nI am still learning.'
				leave_application_json.clear()			
	if not leave_application_json.get('end_date'):
		user_response_end_date = raw_input('Bot: Can you please specify end date of leave duration.\nUser: ')
		context_stack.append('end_date')
		end_date_extracted = date_extractor(user_response_end_date)
		try:
			if end_date_extracted.get('end_date'):
				leave_application_json['end_date'] = end_date_extracted.get('end_date')
			if end_date_extracted.get('start_date') and not leave_application_json.get('start_date'):
					leave_application_json['start_date'] = end_date_extracted.get('start_date')
		except:
			user_response_end_date = raw_input('Bot: Can you please specify end date of leave duration.\nUser: ')
			context_stack.append('end_date')
			end_date_extracted = date_extractor(user_response_end_date)
			try:
				if end_date_extracted.get('end_date'):
					leave_application_json['end_date'] = end_date_extracted.get('end_date')
			except:
				print 'Bot: Sorry for not being able to serve you right now.\nI am still learning.'
				leave_application_json.clear()
	if not leave_application_json.get('out_of_station'):
		user_response_out_of_station = raw_input('Bot: Are you going to be out of station.\nUser: ')
		context_stack.append('yes|no|may be|probably')
		if intent_extractor(user_response_out_of_station) == 'global_yes':
			leave_application_json['out_of_station'] = 'yes'
		elif intent_extractor(user_response_out_of_station) == 'global_no':
			leave_application_json['out_of_station'] = 'no'
		elif intent_extractor(user_response_out_of_station) == 'uncertain':
			leave_application_json['out_of_station'] = 'uncertain'
		else:
			print 'Bot: Sorry for not being able to serve you now'
			leave_application_json.clear()
			return
	try:
		if isinstance(leave_application_json['start_date'],date):
			start_date_date_format = leave_application_json['start_date']
			leave_application_json['start_date'] = str(leave_application_json['start_date'].day)+'-'+str(leave_application_json['start_date'].month)+'-'+str(leave_application_json['start_date'].year)
		else:
			if not isinstance(leave_application_json.get('start_date'),datetime):
				leave_application_json['start_date'] = parser.parse(leave_application_json['start_date']).date()
				start_date_date_format = leave_application_json['start_date']
				leave_application_json['start_date'] = str(leave_application_json['start_date'].day)+'-'+str(leave_application_json['start_date'].month)+'-'+str(leave_application_json['start_date'].year)
			else:
				leave_application_json['start_date'] = leave_application_json['start_date'].date()
				start_date_date_format = leave_application_json['start_date']
				leave_application_json['start_date'] = str(leave_application_json['start_date'].day)+'-'+str(leave_application_json['start_date'].month)+'-'+str(leave_application_json['start_date'].year)
	except:
		print 'Bot: Some issue is there in extracting start date\n'
		leave_application_json.clear()
		return
	try:
		if isinstance(leave_application_json['end_date'],date):
			leave_application_json['end_date'] = leave_application_json['end_date'] + timedelta(days=-1)
			end_date_date_format = leave_application_json['end_date']
			leave_application_json['end_date'] = str(leave_application_json['end_date'].day)+'-'+str(leave_application_json['end_date'].month)+'-'+str(leave_application_json['end_date'].year)
		else:
			if not isinstance(leave_application_json.get('end_date'),datetime):
				leave_application_json['end_date'] = parser.parse(leave_application_json['end_date']).date() + timedelta(days=-1)
				end_date_date_format = leave_application_json['end_date']
				leave_application_json['end_date'] = str(leave_application_json['end_date'].day)+'-'+str(leave_application_json['end_date'].month)+'-'+str(leave_application_json['end_date'].year)
			else:
				leave_application_json['end_date'] = leave_application_json['end_date'].date() + timedelta(days=-1)
				end_date_date_format = leave_application_json['end_date']
				leave_application_json['end_date'] = str(leave_application_json['end_date'].day)+'-'+str(leave_application_json['end_date'].month)+'-'+str(leave_application_json['end_date'].year)
	except:
		print 'Bot: Sorry for it some error in end date extraction.'
		leave_application_json.clear()
		return
		
	if isinstance(end_date_date_format,datetime):
		end_date_date_format = end_date_date_format.date()
	if isinstance(start_date_date_format,datetime):
		start_date_date_format = start_date_date_format.date()
	if end_date_date_format >= start_date_date_format:
		print 'Bot:Your application is leave type as %s and start_date: %s end_date: %s and out_of_station is %s\n'%(leave_application_json['leave_type'],leave_application_json['start_date'],leave_application_json['end_date'],leave_application_json['out_of_station'])
		leave_application_previous = leave_application_json
	else:
		print 'Bot: Some issue is there in start date and end date specified.'
		print 'Bot: Please see that end date is after start date.\n'

	leave_application_json.clear()



if __name__ == '__main__':
	while(1):
		user_response = raw_input('Bot: Hi I am your Leave Application Bot.\nI can help you with booking or updating your Leave Applications.\nUser: ')
		global_intent  = get_global_intent(user_response)
		if global_intent == 'leave_application':
			if intent_extractor(user_response) in ['leave_application','leave_start_date_present','leave_end_date_present']:
				leave_application(user_response)
		# elif global_intent == 'modify_leave_application':
		# 	modify_leave_application(user_response)
		elif global_intent == 'uncertain':
			user_response_1 = raw_input('Bot: Can you please restrict yourself in talking about application or updation of leaves.\nUser: ')
			intent = intent_extractor(user_response_1)
			global_intent  = get_global_intent(user_response_1)
			if global_intent == 'leave_application':
				leave_application(user_response_1)
			# elif global_intent == 'modify_leave_application':
			# 	modify_leave_application(user_response_1)
			elif global_intent == 'uncertain':
				print 'Bot: Sorry for it,As of now I can understand only about leave applications and updations.\n'
			else:
				print 'Bot: Sorry I am trained for understanding about leave applications only.\n'
		elif global_intent == 'greet' or intent_extractor(user_response) == 'greet':
			continue
		else:
			print 'Bot: Sorry I am trained for understanding about leave applications only.\n'
	# while 1:
	# 	print date_extractor(raw_input('Enter: '))


	