from duckling import DucklingWrapper
from datetime import datetime
from datetime import timedelta
import json
from dateutil import parser
import pickle
from nltk.stem.wordnet import WordNetLemmatizer 
import sys
import re
import nltk
import json
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from nltk.classify import Senna
from nltk.stem.wordnet import WordNetLemmatizer
pipeline = Senna('/usr/share/senna-v3.0', ['pos', 'chk', 'ner'])
lmtzr = WordNetLemmatizer()


stop_words = []
file_open_stopwords = open('stop_words.txt','r')
for each_word in file_open_stopwords:
	stop_words.append(each_word.strip())
file_open_stopwords.close()


d = DucklingWrapper()


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

	# print 'tf-idf: ',out_put_class
	# print 'tf-idf: ',passive_tfidf.decision_function(out_put_vector)
	
	out_put_vector_lsa = lsa.transform(out_put_vector)
	# print 'lsa: ',passive_lsa.predict(out_put_vector_lsa)
	# print 'lsa: ',passive_lsa.decision_function(out_put_vector_lsa)
	


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
	leave_dates = {}
	try:
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
						leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
						leave_dates['end_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
				if message_intent in ('leave_start_date_present','leave_end_date_present'):
					text_entities = entity_extractor(message_parsed[0]['text'])
					is_text_duration = is_duration(text_entities)
					date = is_date(text_entities)
					message_entities = entity_extractor(message)
					duration_number_message = is_duration(message_entities)
					if is_text_duration and date:
						if message_intent == 'leave_end_date_present':
							leave_dates['start_date'] = parser.parse(message_parsed[0]['value']['value'].split('T')[0]).date()
							text_parsed = d.parse_time(date)
							if len(text_parsed) == 1:
								leave_dates['end_date'] = parser.parse(text_parsed[0]['value']['value'])
						elif message_intent == 'leave_start_date_present':
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
							leave_dates['start_date'] = leave_dates['end_date'] - timedelta(days=-1*days_to_be_subtracted.days)

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

















		






			

















if __name__ == '__main__':
	while 1:
		print date_extractor(raw_input('Enter: '))



















