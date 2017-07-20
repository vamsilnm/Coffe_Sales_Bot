from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn import svm
import string
punctuations = string.punctuation

from spacy.en import English
parser = English()
#Custom transformer using spaCy 
class predictors(TransformerMixin):
	def transform(self, X, **transform_params):
		return [clean_text(text) for text in X]
	def fit(self, X, y=None, **fit_params):
		return self
	def get_params(self, deep=True):
		return {}

# Basic utility function to clean the text 
def clean_text(text):     
	return text.strip().lower()
def spacy_tokenizer(sentence):
	tokens = parser(sentence)
	tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
	tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]     
	return tokens

vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1)) 
classifier = svm.SVC(decision_function_shape='ovo')

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
	  "coffee"
	 
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

pipe = Pipeline([("cleaner", predictors()),
				 ('vectorizer', vectorizer),
				 ('classifier', classifier)])

pipe.fit([x[0] for x in training_corpus], [x[1] for x in training_corpus]) 


while 1:
	try:
		test_sentence = []
		test_sentence.append(raw_input('Enter: '))
		pred_data = pipe.predict(test_sentence)
		# pred_prob = pipe.decision_function(test_sentence)
		print pred_data
		# print pred_prob
	except Exception,e:
		print e
		continue

