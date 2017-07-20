from textblob.classifiers import NaiveBayesClassifier as NBC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm 
from textblob import TextBlob
import pickle
import json

file_open_data = open('data_2.json','r')
data = json.load(file_open_data)
file_open_data.close()

training_text = []
training_class = []
training_corpus = []
for label in data.keys():
	for text in data[label]["examples"]:
		training_corpus.append((text,label))
		training_class.append(label)
		training_text.append(text)

vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
train_vectors = vectorizer.fit_transform(training_text)

model = NBC(training_corpus)

file_Name = "intent_nb.p"
fileObject = open(file_Name,'wb') 
pickle.dump(model, fileObject)
fileObject.close()

# model = svm.SVC(kernel='rbf')
# model.fit(train_vectors, training_class)  

while 1:
	test_sentence = raw_input('Enter: ')
	# print model.predict(vectorizer.transform([test_sentence]))
	print model.classify(test_sentence)
	prob_dist = model.prob_classify(test_sentence)
	print prob_dist.prob(model.classify(test_sentence)) 

