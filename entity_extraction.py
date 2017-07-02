import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
import json
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle



def sentence_tokens_generation():
	sentence_tokens = []
	while(1):
		sentence = raw_input('Enter: ')
		word_tokens = nltk.word_tokenize(sentence)
		word_tokens_clean = [each_word.strip() for each_word in word_tokens if each_word not in stop]
		word_tokens_pos_tagged = nltk.pos_tag(word_tokens_clean)
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
	while(1):
		sentence = raw_input('Enter: ')
		word_tokens = nltk.word_tokenize(sentence)
		word_tokens_clean = [each_word.strip() for each_word in word_tokens if each_word not in stop]
		word_tokens_pos_tagged = nltk.pos_tag(word_tokens_clean)
		each_sentence_tokens = []
		for each_token in word_tokens_pos_tagged:
			each_sentence_tokens.append(each_token)
		sentence_tokens.append(each_sentence_tokens)

		end = raw_input('do u want to continue: ')
		if end == '0':
			break


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

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]



if __name__ == '__main__':
	# print 'Train sentence genaration'
	# train_sentences = sentence_tokens_generation()
	
	
	
	file_open_train = open('train_sentences.json','r')
	json_file = json.load(file_open_train)
	train_sentences = json_file['train_sentences']

	print 'New Train sentences generation'
	new_train_sentences = sentence_tokens_generation()
	
	train_sentences = train_sentences + new_train_sentences

	print 'Test sentence genaration'
	test_sentences = sentence_tokens_generation_test()
	
	X_train = [sent2features(s) for s in train_sentences]
	y_train = [sent2labels(s) for s in train_sentences]
	file_open_train.close()

	X_test = [sent2features(s) for s in test_sentences]
	# y_test = [sent2labels(s) for s in test_sentences]

	train_sentences_json = {'train_sentences':train_sentences}
	test_sentences_json = {'test_sentences':test_sentences}

	file_open_train = open('train_sentences.json','w')
	json.dump(train_sentences_json,file_open_train)
	file_open_train.close()

	file_open_test = open('test_sentences.json','w')
	json.dump(test_sentences_json,file_open_test)
	file_open_test.close()

	crf = sklearn_crfsuite.CRF(
	    algorithm='lbfgs', 
	    c1=0.1, 
	    c2=0.1, 
	    max_iterations=100, 
	    all_possible_transitions=True
	)
	crf.fit(X_train, y_train)

	file_Name = "entity_model.p"
	fileObject = open(file_Name,'wb') 
	pickle.dump(crf, fileObject)
	fileObject.close()


	y_pred = crf.predict(X_test)

	
	for each_sentence in zip(test_sentences,y_pred):
		for each_word in zip(each_sentence[0],each_sentence[1]):
			print each_word[0][0],each_word[1]
		print '\n'

