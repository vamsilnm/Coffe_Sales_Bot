from duckling import DucklingWrapper
from datetime import datetime
from datetime import timedelta
import json
from dateutil import parser
import pickle
from nltk.stem.wordnet import WordNetLemmatizer 
import sys
import datefinder
import re
from natty import DateParser

d = DucklingWrapper()

lmtzr = WordNetLemmatizer()
#Stopwords generation
stop_words = []
file_open_stopwords = open('stop_words.txt','r')
for each_word in file_open_stopwords:
	stop_words.append(each_word.strip())
file_open_stopwords.close()


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




def date_extractor(message):
	leave_dates = {}
	try:
		message_parsed = d.parse_time(message)
		if len(message_parsed):
			if not message_parsed[0]['value'].get('grain'):
				if message_parsed[0]['value']['value'].get('to') and message_parsed[0]['value']['value'].get('from'):
					end_date_string = message_parsed[0]['value']['value'].get('to').split('T')[0]
					leave_dates['end_date'] = end_date_string
					start_date_string = message_parsed[0]['value']['value'].get('from').split('T')[0]
					leave_dates['start_date'] = start_date_string
			elif len(message_parsed):
				if len(message_parsed) == 1:
					leave_dates['start_date'] = message_parsed[0]['value']['value'].split('T')[0]
					leave_dates['end_date'] = message_parsed[0]['value']['value'].split('T')[0]
				# elif len(message_parsed) > 1 and not len(matches_list_total):
				# 	probable_dates = []
				# 	is_hour_day = 0
				# 	for each_item in message_parsed:
				# 		if each_item['value']['grain'] == 'day' or each_item['value']['grain'] == 'second':
				# 			probable_dates.append(parser.parse(each_item['value']['value']))
				# 	if len(probable_dates) == 2:
				# 		if probable_dates[0] < probable_dates[1]:
				# 			leave_dates['start_date'] = str(probable_dates[0].year)+'-'+str(probable_dates[0].month)+'-'+str(probable_dates[0].day)
				# 			leave_dates['end_date'] = str(probable_dates[1].year)+'-'+str(probable_dates[1].month)+'-'+str(probable_dates[1].day)
				# 		else:
				# 			leave_dates['start_date'] = str(probable_dates[1].year)+'-'+str(probable_dates[1].month)+'-'+str(probable_dates[1].day)
				# 			leave_dates['end_date'] = str(probable_dates[0].year)+'-'+str(probable_dates[0].month)+'-'+str(probable_dates[0].day)
				if not leave_dates.get('start_date') and not leave_dates.get('end_date') :
					days_to_be_added = 0
					days_to_be_subtracted = 0
					parsed_message_keys = {}
					grain_hour_count = 0
					grain_day_count = 0
					if intent_extractor(message) == 'leave_start_date_present':
						if len(message_parsed) == 2:
							for each_message_parsed in message_parsed:
								if each_message_parsed['value']['grain'] == 'hour':
									grain_hour_count += 1
								elif each_message_parsed['value']['grain'] == 'day':
									grain_day_count += 1
							if grain_day_count == 1 and grain_hour_count == 1:
								if message_parsed[0]['value']['grain'] == 'day' :
									days_to_be_added = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
									leave_dates['start_date'] = message_parsed[1]['value']['value'].split('T')[0]
								else:
									leave_dates['start_date'] = message_parsed[0]['value']['value'].split('T')[0]
									days_to_be_added = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
							elif grain_hour_count == 2:
								first_text = message_parsed[0]['text']
								second_text = message_parsed[1]['text']
								if len(first_text) > len(second_text):
									days_to_be_subtracted = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
									leave_dates['end_date'] = message_parsed[0]['value']['value'].split('T')[0]
								else:
									days_to_be_subtracted = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
									leave_dates['end_date'] = message_parsed[1]['value']['value'].split('T')[0]
							elif grain_day_count == 2:
								first_text_is_date = 0
								second_text_is_date = 0
								first_text = message_parsed[0]['text']
								second_text = message_parsed[1]['text']
								first_text_match = datefinder.find_dates(first_text)
								first_text_match_list = [match for match in first_text_match]
								if len(first_text_match_list):
									first_text_is_date = 1
								second_text_match = datefinder.find_dates(second_text)
								second_text_match_list = [match for match in second_text_match]
								if len(second_text_match_list):
									second_text_is_date = 1
								if first_text_is_date and not second_text_is_date:
									leave_dates['start_date'] = message_parsed[0]['value']['value'].split('T')[0]
									days_to_be_added = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
								elif not first_text_is_date and second_text_is_date:
									leave_dates['start_date'] = message_parsed[1]['value']['value'].split('T')[0]
									days_to_be_added = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
						if days_to_be_added and leave_dates.get('start_date'):
							leave_dates['end_date'] = parser.parse(leave_dates['start_date']) + timedelta(days=days_to_be_added)
						elif days_to_be_subtracted and leave_dates.get('end_date'):
							leave_dates['start_date'] = parser.parse(leave_dates['end_date']) + timedelta(days= -1*days_to_be_subtracted)
						else:
							leave_dates.clear()
					elif intent_extractor(message) == 'leave_end_date_present':
						if len(message_parsed) == 2:
							for each_message_parsed in message_parsed:
								if each_message_parsed['value']['grain'] == 'hour':
									grain_hour_count += 1
								elif each_message_parsed['value']['grain'] == 'day':
									grain_day_count += 1
							if grain_day_count == 1 and grain_hour_count == 1:
								if message_parsed[0]['value']['grain'] == 'day' :
									days_to_be_subtracted = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
									leave_dates['end_date'] = message_parsed[1]['value']['value'].split('T')[0]
								else:
									days_to_be_subtracted = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
									leave_dates['end_date'] = message_parsed[0]['value']['value'].split('T')[0]
							elif grain_hour_count == 2:
								first_text = message_parsed[0]['text']
								second_text = message_parsed[1]['text']
								if len(first_text) > len(second_text):
									days_to_be_added = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
									leave_dates['start_date'] = message_parsed[0]['value']['value'].split('T')[0]
								else:
									days_to_be_added = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
									leave_dates['start_date'] = message_parsed[1]['value']['value'].split('T')[0]
							elif grain_day_count == 2:
								first_text_is_date = 0
								second_text_is_date = 0
								first_text = message_parsed[0]['text']
								second_text = message_parsed[1]['text']
								first_text_match = datefinder.find_dates(first_text)
								first_text_match_list = [match for match in first_text_match]
								if len(first_text_match_list):
									first_text_is_date = 1
								second_text_match = datefinder.find_dates(second_text)
								second_text_match_list = [match for match in second_text_match]
								if len(second_text_match_list):
									second_text_is_date = 1
								if first_text_is_date and not second_text_is_date:
									leave_dates['end_date'] = message_parsed[0]['value']['value'].split('T')[0]
									days_to_be_subtracted = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
								elif not first_text_is_date and second_text_is_date:
									leave_dates['end_date'] = message_parsed[1]['value']['value'].split('T')[0]
									days_to_be_subtracted = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
						if days_to_be_added and leave_dates.get('start_date'):
							leave_dates['end_date'] = parser.parse(leave_dates['start_date']) + timedelta(days=days_to_be_added)
						elif days_to_be_subtracted and leave_dates.get('end_date'):
							leave_dates['start_date'] = parser.parse(leave_dates['end_date']) + timedelta(days= -1*days_to_be_subtracted)
						else:
							leave_dates.clear()
					if intent_extractor(message) and not len(leave_dates):
						invalid_format_duration = 0
						for each_message_parsed in message_parsed:
							message_containing_date = each_message_parsed['text']
							matches = datefinder.find_dates(message_containing_date)
							matches_list = [match for match in matches]
							if len(matches_list) == 1:
								if intent_extractor(message) == 'leave_end_date_present':
									leave_dates['end_date'] = matches_list[0]
									leave_dates['start_date'] = each_message_parsed['value']['value']
								elif intent_extractor(message) == 'leave_start_date_present':
									leave_dates['start_date'] = matches_list[0]
									leave_dates['end_date'] = each_message_parsed['value']['value']
						if not len(leave_dates):
							probable_dates = []
							probable_days = 0
							for each_date in re.finditer(r'(\d+/\d+/\d+)', message):
								probable_dates.append(datetime.strptime(each_date.group(1), "%d/%m/%Y"))
							if len(probable_dates) == 2:
								leave_dates['start_date'] = probable_dates[0]
								leave_dates['end_date'] = probable_dates[1]
							elif len(probable_dates) == 1:
								if intent_extractor(message) == 'only_one_date_present':
									leave_dates['start_date'] = probable_dates[0]
									leave_dates['end_date'] = probable_dates[0]
								else:
									for each_day in re.finditer(r'(\d\s|\d\w+\s)', message):
										probable_days = int(re.search(r'\d',each_day.group(1)).group(0))
									is_days = 0
									for each_word in message.split():
										each_word_match = re.search(r'[a-z]+',each_word.lower().strip())
										if each_word_match:
											each_word = each_word_match.group(0)
											if lmtzr.lemmatize(each_word.strip().lower()) == 'day':
												is_days = 1
									if probable_days:
										if intent_extractor(message) == 'leave_start_date_present':
											leave_dates['start_date'] = probable_dates[0]
											if is_days:
												leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days=probable_days)
											else:
												print 'Please enter only in days'
												invalid_format_duration = 1
												leave_dates.clear()
										elif intent_extractor(message) == 'leave_end_date_present':
											leave_dates['end_date'] = probable_dates[0]
											if is_days:
												leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days=-1*probable_days)
											else:
												print 'Please enter only in days'
												invalid_format_duration = 1
												leave_dates.clear()

						if not len(leave_dates) and not invalid_format_duration:
							probable_dates = []
							probable_days = 0
							for each_date in re.finditer(r'(\d+-\d+-\d+)', message):
								probable_dates.append(datetime.strptime(each_date.group(1), "%d-%m-%Y"))
							if len(probable_dates) == 2:
								leave_dates['start_date'] = probable_dates[0]
								leave_dates['end_date'] = probable_dates[1]
							elif len(probable_dates) == 1:
								if intent_extractor(message) == 'only_one_date_present':
									leave_dates['start_date'] = probable_dates[0]
									leave_dates['end_date'] = probable_dates[0]
								else:
									for each_day in re.finditer(r'(\d\s|\d\w+\s)', message):
										probable_days = int(re.search(r'\d',each_day.group(1)).group(0))
									is_days = 0
									for each_word in message.split():
										each_word_match = re.search(r'[a-z]+',each_word.lower().strip())
										if each_word_match:
											each_word = each_word_match.group(0)
											if lmtzr.lemmatize(each_word.strip().lower()) == 'day':
												is_days = 1
									if probable_days:
										if intent_extractor(message) == 'leave_start_date_present':
											leave_dates['start_date'] = probable_dates[0]
											if is_days:
												leave_dates['end_date'] = leave_dates['start_date'] + timedelta(days=probable_days)
											else:
												print 'Please enter only in days'
												leave_dates.clear()
										elif intent_extractor(message) == 'leave_end_date_present':
											leave_dates['end_date'] = probable_dates[0]
											if is_days:
												leave_dates['start_date'] = leave_dates['end_date'] + timedelta(days=-1*probable_days)
											else:
												print 'Please enter only in days'
												invalid_format_duration = 1
												leave_dates.clear()								
					elif not len(leave_dates):
						probable_dates = []
						for each_date in re.finditer(r'(\d+/\d+/\d+)', message):
							probable_dates.append(datetime.strptime(each_date.group(1), "%d/%m/%Y"))
						if len(probable_dates) == 2:
							leave_dates['start_date'] = probable_dates[0]
							leave_dates['end_date'] = probable_dates[1]
						else:
							probable_dates = []
							for each_date in re.finditer(r'(\d+-\d+-\d+)', message):
								probable_dates.append(datetime.strptime(each_date.group(1), "%d-%m-%Y"))
								if len(probable_dates) == 2:
									leave_dates['start_date'] = probable_dates[0]
									leave_dates['end_date'] = probable_dates[1]
					# elif not len(leave_dates):
					# 	print 'Date Not Found'
					# 	print message_parsed
						
						# date_extractor(raw_input('Enter date in more simple way\n: '))
			if leave_dates.get('start_date') and leave_dates.get('end_date'):
				is_month = 0
				is_year = 0
				if not isinstance(leave_dates.get('start_date'),datetime):
					leave_dates['start_date'] = parser.parse(leave_dates['start_date'])
				if not isinstance(leave_dates.get('end_date'),datetime):
					leave_dates['end_date'] = parser.parse(leave_dates['end_date'])
				if leave_dates['start_date'].date() >= datetime.today().date() and leave_dates['end_date'].date() >= datetime.today().date():
					if leave_dates['start_date'].date() == leave_dates['end_date'].date():
						for each_word in message.split():
							if each_word.strip().lower() == 'month' or each_word.strip().lower() == 'months':
								is_month = 1
								break
							if each_word.strip().lower() == 'years' or each_word.strip().lower() == 'years':
								is_year = 1
								break
					if not is_month and not is_year:
						leave_dates['start_date'] = str(leave_dates['start_date'].year)+'-'+str(leave_dates['start_date'].month)+'-'+str(leave_dates['start_date'].day)
						leave_dates['end_date'] = str(leave_dates['end_date'].year)+'-'+str(leave_dates['end_date'].month)+'-'+str(leave_dates['end_date'].day)
						return leave_dates
				else:
					leave_dates.clear()
					return leave_dates
		else:
			return leave_dates
	except Exception,e:
		exc_type, exc_obj, exc_tb = sys.exc_info()
		print (exc_type,exc_tb.tb_lineno)
		print message_parsed
		print 'Exception',e



def test(message):
	message_parsed = d.parse_time(message)
	leave_dates = {}
	# if not leave_dates.get('start_date') and not leave_dates.get('end_date') :
	days_to_be_added = 0
	days_to_be_subtracted = 0
	grain_hour_count = 0
	grain_day_count = 0
	if intent_extractor(message) == 'leave_start_date_present':
		if len(message_parsed) == 2:
			for each_message_parsed in message_parsed:
				if each_message_parsed['value']['grain'] == 'hour':
					grain_hour_count += 1
				elif each_message_parsed['value']['grain'] == 'day':
					grain_day_count += 1
			if grain_day_count == 1 and grain_hour_count == 1:
				if message_parsed[0]['value']['grain'] == 'day' :
					days_to_be_added = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
					leave_dates['start_date'] = message_parsed[1]['value']['value'].split('T')[0]
				else:
					leave_dates['start_date'] = message_parsed[0]['value']['value'].split('T')[0]
					days_to_be_added = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
			elif grain_hour_count == 2:
				first_text = message_parsed[0]['text']
				second_text = message_parsed[1]['text']
				if len(first_text) > len(second_text):
					days_to_be_subtracted = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
					leave_dates['end_date'] = message_parsed[0]['value']['value'].split('T')[0]
				else:
					days_to_be_subtracted = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
					leave_dates['end_date'] = message_parsed[1]['value']['value'].split('T')[0]
			elif grain_day_count == 2:
				first_text_is_date = 0
				second_text_is_date = 0
				first_text = message_parsed[0]['text']
				second_text = message_parsed[1]['text']
				first_text_match = datefinder.find_dates(first_text)
				first_text_match_list = [match for match in first_text_match]
				if len(first_text_match_list):
					first_text_is_date = 1
				second_text_match = datefinder.find_dates(second_text)
				second_text_match_list = [match for match in second_text_match]
				if len(second_text_match_list):
					second_text_is_date = 1
				if first_text_is_date and not second_text_is_date:
					leave_dates['start_date'] = message_parsed[0]['value']['value'].split('T')[0]
					days_to_be_added = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
				elif not first_text_is_date and second_text_is_date:
					leave_dates['start_date'] = message_parsed[1]['value']['value'].split('T')[0]
					days_to_be_added = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
		if days_to_be_added and leave_dates.get('start_date'):
			leave_dates['end_date'] = parser.parse(leave_dates['start_date']) + timedelta(days=days_to_be_added)
		elif days_to_be_subtracted and leave_dates.get('end_date'):
			leave_dates['start_date'] = parser.parse(leave_dates['end_date']) + timedelta(days= -1*days_to_be_subtracted)
		else:
			leave_dates.clear()
	elif intent_extractor(message) == 'leave_end_date_present':
		if len(message_parsed) == 2:
			for each_message_parsed in message_parsed:
				if each_message_parsed['value']['grain'] == 'hour':
					grain_hour_count += 1
				elif each_message_parsed['value']['grain'] == 'day':
					grain_day_count += 1
			if grain_day_count == 1 and grain_hour_count == 1:
				if message_parsed[0]['value']['grain'] == 'day' :
					days_to_be_subtracted = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
					leave_dates['end_date'] = message_parsed[1]['value']['value'].split('T')[0]
				else:
					days_to_be_subtracted = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
					leave_dates['end_date'] = message_parsed[0]['value']['value'].split('T')[0]
			elif grain_hour_count == 2:
				first_text = message_parsed[0]['text']
				second_text = message_parsed[1]['text']
				if len(first_text) > len(second_text):
					days_to_be_added = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
					leave_dates['start_date'] = message_parsed[0]['value']['value'].split('T')[0]
				else:
					days_to_be_added = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
					leave_dates['start_date'] = message_parsed[1]['value']['value'].split('T')[0]
			elif grain_day_count == 2:
				first_text_is_date = 0
				second_text_is_date = 0
				first_text = message_parsed[0]['text']
				second_text = message_parsed[1]['text']
				first_text_match = datefinder.find_dates(first_text)
				first_text_match_list = [match for match in first_text_match]
				if len(first_text_match_list):
					first_text_is_date = 1
				second_text_match = datefinder.find_dates(second_text)
				second_text_match_list = [match for match in second_text_match]
				if len(second_text_match_list):
					second_text_is_date = 1
				if first_text_is_date and not second_text_is_date:
					leave_dates['end_date'] = message_parsed[0]['value']['value'].split('T')[0]
					days_to_be_subtracted = (parser.parse(message_parsed[1]['value']['value']).date() - datetime.today().date()).days
				elif not first_text_is_date and second_text_is_date:
					leave_dates['end_date'] = message_parsed[1]['value']['value'].split('T')[0]
					days_to_be_subtracted = (parser.parse(message_parsed[0]['value']['value']).date() - datetime.today().date()).days
		if days_to_be_added and leave_dates.get('start_date'):
			leave_dates['end_date'] = parser.parse(leave_dates['start_date']) + timedelta(days=days_to_be_added)
		elif days_to_be_subtracted and leave_dates.get('end_date'):
			leave_dates['start_date'] = parser.parse(leave_dates['end_date']) + timedelta(days= -1*days_to_be_subtracted)
		else:
			leave_dates.clear()
		return leave_dates
	else:
		print 'Intent Wrong'
		return leave_dates



if __name__ == '__main__':
	# while 1:
	# 	tfidf_class,lsa_class = intent_extractor(raw_input('Enter: '))
	# 	print 'tfidf_class',tfidf_class
	# 	print 'lsa_class',lsa_class
	# while 1:
	# 	print test(raw_input('Enter: '))
	while 1:
		print date_extractor(raw_input('Enter: '))