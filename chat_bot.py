import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import json
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

vectorizer = TfidfVectorizer(stop_words='english',analyzer='word',lowercase=True,sublinear_tf=True)

order_json = {}
data={
	"global_no": {
	"examples" : ["nope","no never","nah!","never","no","a big no","always a no","reject"]
	
  },
	"global_yes": {
	"examples" : ["yes","yaa","go ahead","yup","yeah"]
	
  },
  "greet": {
	"examples" : ["hello","hey there","howdy","hello","hi","hey","hey ho"]
	
  },
  "beverage": {
	"examples" : [
	  "i would like to have coffee",
	  "i would like to have coffee with soy milk",
	  "can you please get me a coffee with almond milk",
	  "can you please get me a coffee",
	  "i need a coffee with soy milk",
	  "maybe coffee",
	  "coffee it is!!!!",
	  "i want coffee",
	  "can you please get me a coffee",
	  "i love to have coffee",
	  "i will go with coffee this time",
	  "coffee",
	  "need a coffee",
	  "need to order a cup coffee",
	  "can i get a coffee",
	  "get me a coffee",
	  "get me a coffee please",
	  "i would like to have cappuccino with soy milk",
	  "i will go with cappuccino this time",
	  "i like to go with cappuccino and soy milk",
	  "i want cappuccino with almond milk",
	  "i need a cappuccino with almond milk",
	  "can you please get me a cappuucino with soy milk",
	  "can you get me a cappuccino",
	  "nothing can beat cappuccino",
	  "i love to have cappuccino"
	  "cappuccino",
	  "can you get me a cappuccino",
	  "please get me a cappuccino",
	  "can you please get me a cappuccino",
	  "i like to have cappuccino",
	  "need to order a cappuccino",
	  "please can i have cappuccino",
	  "i like to have cappuccino",
	  "i want to have cappuccino"
	]
  },
  "update": {
	"examples": [
	   "actually go with soy milk",
	   "can you change it to coffee",
	   "can you change it to cappuccino",
	   "no i changed my mind.please go with coffee",
	   "no i changed my mind.please go with cappuccino",
	   "update it with almond milk",
	   "change it to almond milk",
	   "change it to soy milk",
	   "i want to update",
	   "can you please update my order",
	   "i want to change my order",
	   "updte with coffee",
	   "i want to update my order",
	   "i want to change it",
	   "update it with soy milk"
	]

  }
}

training_text = []

for label in data.keys():
	for text in data[label]["examples"]:
		training_text.append(text)



_ = vectorizer.fit_transform(training_text)

def intent_extractor(message):
	intent_model = open('classif.p','rb')
	clf = pickle.load(intent_model)
	intent_model.close()
	out_put = []
	out_put.append(message.lower())
	out_put_vector = vectorizer.transform(out_put)
	out_put_class = clf.predict(out_put_vector)
	return out_put_class[0],list(clf.decision_function(out_put_vector)[0])

def sentence_tokens_generation(message):
	sentence_tokens = []
	sentence = message
	word_tokens = nltk.word_tokenize(sentence)
	word_tokens_clean = [each_word.strip() for each_word in word_tokens if each_word not in stop]
	word_tokens_pos_tagged = nltk.pos_tag(word_tokens_clean)
	each_sentence_tokens = []
	for each_token in word_tokens_pos_tagged:
		each_sentence_tokens.append(each_token)
	sentence_tokens.append(each_sentence_tokens)
	return sentence_tokens

def word2features(sent, i):
	word = sent[i][0]
	postag = sent[i][1]
	
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

def sent2features(sent):
	return [word2features(sent, i) for i in range(len(sent))]

def entity_extractor(message):
	file_open_pickle = open('entity_model.p','rb')
	crf = pickle.load(file_open_pickle) 
	
	message_tokens = sentence_tokens_generation(message)
	message_vector = [sent2features(s) for s in message_tokens]
	message_entity = crf.predict(message_vector)

	entity_list = []
	for each_sentence in zip(message_tokens,message_entity):
		for each_word in zip(each_sentence[0],each_sentence[1]):
			entity_list.append((each_word[0][0],each_word[1]))

	return entity_list

def beverage_extractor(message):
	entities = entity_extractor(message)
	beverages = []
	if entities:
		for each_entity in entities:
			if each_entity[1] == 'b':
				beverages.append(each_entity[0])
		return beverages
	else:
		return None

def milk_extractor(message):
	entities = entity_extractor(message)
	milks = []
	if entities:
		for each_entity in entities:
			if each_entity[1] == 'm':
				milks.append(each_entity[0])
		return milks
	else:
		return None
def assertion_intent(message):
	intent_choice,intent_choice_confidence = intent_extractor(message)
	if any(score>0 for score in intent_choice_confidence):
		if intent_choice == 'global_yes':
			return 1
		else:
			return 0
	else:
		if intent_choice == 'global_yes':
			return 0
		elif intent_choice == 'global_no':
			return 1
		else:
			return 0

def order(message):
	global order_json
	previous_order_beverage = order_json['beverage'] if order_json.get('beverage') else None
	previous_order_milk = order_json['milk'] if order_json.get('milk') else None
	order_json.clear()
	if beverage_extractor(message):
		order_json['beverage'] = beverage_extractor(message)
	if milk_extractor(message):
		order_json['milk'] = milk_extractor(message)

	if not order_json.get('beverage'):
		user_beverage = raw_input('Bot: Can you please type for which beverage you are looking.\nUser: ')
		if beverage_extractor(user_beverage):
			order_json['beverage'] = beverage_extractor(user_beverage)
		else:
			print 'Bot: Sorry for not being able to understand you.I am still learning.'
	if not order_json.get('milk'):
		user_milk = raw_input('Bot: Can you please type with which milk you want the beverage\nUser: ')
		if milk_extractor(user_milk):
			order_json['milk'] = milk_extractor(user_milk)
		else:
			print 'Bot: Sorry for not being able to understand you.I am still learning.'
	if order_json.get('beverage') and order_json.get('milk'):
		order_string = ''
		milk_index = 0
		if any(beverage_type.lower().strip() == 'black' for beverage_type in order_json['beverage']):
			is_black = 1
			black_index = order_json['beverage'].index('black')
		else:
			is_black = 0
		if not is_black:
			for each_element in order_json['beverage']:
				try:
					order_string += each_element + ' with '+  order_json['milk'][milk_index]+' milk. '
					milk_index += 1
				except:
					order_string += each_element + ' '+'. '
		elif is_black:
			for each_element in order_json['beverage']:
				if each_element.lower().strip() != 'black':
					if order_json['beverage'].index(each_element) != black_index +1:
						try:
							order_string += each_element + ' with '+  order_json['milk'][milk_index]+' milk. '
							milk_index += 1
						except:
							order_string += each_element + '. '
				elif each_element.lower().strip() == 'black':
					try:
						order_string += each_element +' '+order_json['beverage'][black_index+1] + ' with '+  order_json['milk'][milk_index] +' milk. '
						milk_index += 1
					except:
						order_string += each_element +' '+order_json['beverage'][black_index+1] + '.'
		confirm = raw_input('Bot:Your order is %sDo you want to confirm it.\nUser: '%(order_string))
		if assertion_intent(confirm):
			print 'Bot: Here is your order %s\n'%(order_string)
		else:
			order_json.clear()
			order_json['milk'] = previous_order_milk
			order_json['beverage'] = previous_order_beverage
			is_update = raw_input('Bot: Do you want to change or update your above order.\nUser: ')
			if assertion_intent(is_update):
				update(is_update)
			else:
				intent_is_update,_ = intent_extractor(is_update)
				if intent_is_update =='update':
					update(is_update) 
				else:
					print 'Bot: Thanks for being here.Hope you had a nice time.'
	else:
		print 'Bot: Thanks for being here.Hope you had a nice time.'


def update(message):
	is_success = 1
	if order_json.get('beverage') and order_json.get('milk'):
		if beverage_extractor(message) or milk_extractor(message):
			order_json['beverage'] = beverage_extractor(message) if beverage_extractor(message) else order_json['beverage']
			order_json['milk'] = milk_extractor(message) if milk_extractor(message) else order_json['milk']
		else:
			update_order = raw_input('Bot: Please enter what do you want to update.\nUser: ')
			if beverage_extractor(update_order) and milk_extractor(update_order):
				order_json['beverage'] = beverage_extractor(update_order)
				order_json['milk'] = milk_extractor(update_order)
			elif milk_extractor(update_order):
				order_json['milk'] = milk_extractor(update_order)
				beverage_update = raw_input('Bot: Do you want to update beverage type.\nUser: ')
				if assertion_intent(beverage_update):
					order_json['beverage'] = beverage_extractor(raw_input('Bot: Please enter the beverage type you want to be updated.\nUser: '))
			elif beverage_extractor(update_order):
				order_json['beverage'] = beverage_extractor(update_order)
				milk_update = raw_input('Bot: Do you want to update milk type.\nUser: ')
				if assertion_intent(milk_update):
					order_json['milk'] = milk_extractor(raw_input('Bot: Please enter the milk type you want to be updated.\nUser: '))
			else:
				is_success = 0
				print 'Bot: Sorry for not being able to serve you.'
		if is_success:
			order_string = ''
			milk_index = 0
			order_string = ' '.join(order_json['beverage'])+' ' + order_json['milk'][0]+' milk.'
			print 'Bot: Here is your updated order %s\n'%(order_string)
	else:
		print 'Bot: Nothing to update\n'




if __name__ == '__main__':
	while(1):
		user_response = raw_input('Bot: Hi I am your Cafe Coffee Day Assistant.\nI can help you with booking or updating your beverage orders.\nUser: ')
		intent,intent_confidence = intent_extractor(user_response)
		if any(score>0 for score in intent_confidence):
			if intent == 'beverage':
				order(user_response)
			elif intent == 'update':
				update(user_response)
			elif intent == 'greet':
				user_response = raw_input('Bot:I can help you to try our exiting beverages with our customised milks\nUser: ')
				intent,_ = intent_extractor(user_response)
				if intent == 'beverage':
					order(user_response)
				elif intent == 'update':
					update(user_response)
				else:
					print 'Bot: Sorry for not being able to server you.I am still learning'
			else:
				user_response = raw_input('Bot: Sorry for that.Can you let me know what you are looking for.\nUser:')
				intent,_ = intent_extractor(user_response)
				if intent == 'beverage':
					order(user_response)
				elif intent == 'update':
					update(user_response)
				else:
					print 'Bot: Sorry for not being able to server you.I am still learning'
		else:
			print 'Bot:Can we have a chat regarding Coffee itself\n'
		



		


	












