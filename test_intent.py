from textblob.classifiers import NaiveBayesClassifier as NBC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm 
from textblob import TextBlob
import pickle

data={
	"global_no": {
	"examples" : ["nope","no never","nah!","never","a big no","always a no","reject"]
	
  },
	"global_yes": {
	"examples" : ["yes","yaa","go ahead","yup","yeah","ya","a big yes"]
	
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
	  "can i get a coffee"
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

