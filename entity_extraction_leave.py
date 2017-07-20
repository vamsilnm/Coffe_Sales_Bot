import nltk
import json
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle
from nltk.classify import Senna
from nltk.stem.wordnet import WordNetLemmatizer

pipeline = Senna('/usr/share/senna-v3.0', ['pos', 'chk', 'ner'])
lmtzr = WordNetLemmatizer()


stop_words = []
file_open_stopwords = open('stop_words.txt','r')
for each_word in file_open_stopwords:
	stop_words.append(each_word.strip())
file_open_stopwords.close()

def sentence_tokens_generation():
	sentence_tokens = []
	while(1):
		sentence = raw_input('Enter: ')
		word_tokens = sentence.split()
		word_tokens_clean = [each_word.strip() for each_word in word_tokens]
		word_tokens_pos_tagged_containing_stopwords = [(token['word'], token['chk'], token['ner'], token['pos']) for token in pipeline.tag(word_tokens_clean)]
		word_tokens_pos_tagged = [word for word in word_tokens_pos_tagged_containing_stopwords if word[0] not in stop_words]
		each_sentence_tokens = []
		for each_token in word_tokens_pos_tagged:
			each_sentence_tokens.append(each_token + (raw_input('Entity for %s: '%(each_token[0])),))
		sentence_tokens.append(each_sentence_tokens)

		end = raw_input('do u want to continue: ')
		if end == '0':
			break



	return sentence_tokens
def sentence_tokens_generation_test():
	sentence_tokens = []
	sentence = raw_input('Enter: ')
	word_tokens = sentence.split()
	word_tokens_clean = [each_word.strip() for each_word in word_tokens]
	word_tokens_pos_tagged_containing_stopwords = [(token['word'], token['chk'], token['ner'], token['pos']) for token in pipeline.tag(word_tokens_clean)]
	word_tokens_pos_tagged = [word for word in word_tokens_pos_tagged_containing_stopwords if word[0] not in stop_words]	
	each_sentence_tokens = []
	for each_token in word_tokens_pos_tagged:
		each_sentence_tokens.append(each_token)
	sentence_tokens.append(each_sentence_tokens)

	return sentence_tokens

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


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, chunk, ner_tag, postag, label in sent]

def sent2tokens(sent):
    return [token for token, chunk, ner_tag, postag, label in sent]



if __name__ == '__main__':
	choice = raw_input("Enter 1 if you are training from scratch\n2 if you want to add more data for training and\n3 for testing: ")
	if choice == '1':
		print 'Train sentence genaration'
		train_sentences = sentence_tokens_generation()
		X_train = [sent2features(s) for s in train_sentences]
		y_train = [sent2labels(s) for s in train_sentences]
		train_sentences_json = {'train_sentences':train_sentences}
		file_open_train = open('train_sentences.json','w')
		json.dump(train_sentences_json,file_open_train)
		file_open_train.close()

		crf = sklearn_crfsuite.CRF(
		    algorithm='lbfgs', 
		    c1=0.1, 
		    c2=0.1, 
		    max_iterations=100, 
		    all_possible_transitions=True
		)
		crf.fit(X_train, y_train)

		file_Name = "entity_model_leave.p"
		fileObject = open(file_Name,'wb') 
		pickle.dump(crf, fileObject)
		fileObject.close()

		while 1:
			print 'Testing'
			test_sentences = sentence_tokens_generation_test()
			X_test = [sent2features(s) for s in test_sentences]
			y_pred = crf.predict(X_test)
			for each_sentence in zip(test_sentences,y_pred):
				for each_word in zip(each_sentence[0],each_sentence[1]):
					print each_word[0][0],each_word[1]
				print '\n'
			end = raw_input('Do you want to continue: ')
			if end == '0':
				break

	elif choice == '2':
		file_open_train = open('train_sentences.json','r')
		json_file = json.load(file_open_train)
		file_open_train.close()
		train_sentences = json_file['train_sentences']

		print 'New Train sentences generation'
		new_train_sentences = sentence_tokens_generation()
		
		train_sentences = train_sentences + new_train_sentences
		X_train = [sent2features(s) for s in train_sentences]
		y_train = [sent2labels(s) for s in train_sentences]
		train_sentences_json = {'train_sentences':train_sentences}
		
		file_open_train = open('train_sentences.json','w')
		json.dump(train_sentences_json,file_open_train)
		file_open_train.close()

		crf = sklearn_crfsuite.CRF(
		    algorithm='lbfgs', 
		    c1=0.1, 
		    c2=0.1, 
		    max_iterations=100, 
		    all_possible_transitions=True
		)
		crf.fit(X_train, y_train)

		file_Name = "entity_model_leave.p"
		fileObject = open(file_Name,'wb') 
		pickle.dump(crf, fileObject)
		fileObject.close()

		while 1:
			print 'Testing'
			test_sentences = sentence_tokens_generation_test()
			X_test = [sent2features(s) for s in test_sentences]
			y_pred = crf.predict(X_test)
			for each_sentence in zip(test_sentences,y_pred):
				for each_word in zip(each_sentence[0],each_sentence[1]):
					print each_word[0][0],each_word[1]
				print '\n'
			end = raw_input('Do you want to continue: ')
			if end == '0':
				break
	elif choice == '3':
		file_open_train = open('train_sentences.json','r')
		json_file = json.load(file_open_train)
		file_open_train.close()
		train_sentences = json_file['train_sentences']
		X_train = [sent2features(s) for s in train_sentences]
		y_train = [sent2labels(s) for s in train_sentences]

		crf = sklearn_crfsuite.CRF(
		    algorithm='lbfgs', 
		    c1=0.1, 
		    c2=0.1, 
		    max_iterations=100, 
		    all_possible_transitions=True
		)
		crf.fit(X_train, y_train)

		file_Name = "entity_model_leave.p"
		fileObject = open(file_Name,'wb') 
		pickle.dump(crf, fileObject)
		fileObject.close()

		while 1:
			print 'Testing'
			test_sentences = sentence_tokens_generation_test()
			X_test = [sent2features(s) for s in test_sentences]
			y_pred = crf.predict(X_test)
			for each_sentence in zip(test_sentences,y_pred):
				for each_word in zip(each_sentence[0],each_sentence[1]):
					print each_word[0][0],each_word[1]
				print '\n'
			end = raw_input('Do you want to continue: ')
			if end == '0':
				break

	