import nltk
import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
from sklearn.pipeline import Pipeline
from collections import Counter, defaultdict
from sklearn.externals import joblib


data={
	"global_no": {
	"examples" : ["nope","no never","nah!","never","a big no","always a no","reject"]
	
  },
	"global_yes": {
	"examples" : ["yes","yaa","go ahead","yup","yeah","ya"]
	
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
training_class = []

#Stop_words Generation
stop_words = []
file_open = open('stop_words.txt','r')
for each_word in file_open:
	stop_words.append(each_word.strip())


for label in data.keys():
	for text in data[label]["examples"]:
		training_class.append(label)
		text_clean = [each_word for each_word in nltk.word_tokenize(text) if each_word not in stop_words] 
		training_text.append(text_clean)

X,y = np.array(training_text),np.array(training_class)



GLOVE_6B_50D_PATH = '/media/vamsi/Education/glove.840B.300d/glove.840B.300d.txt'

with open(GLOVE_6B_50D_PATH, "rb") as lines:
	word2vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
			   for line in lines}


glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_50D_PATH, "rb") as infile:
	for line in infile:
		parts = line.split()
		word = parts[0]
		nums = map(float, parts[1:])
		if word in all_words:
			glove_small[word] = np.array(nums)


model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.index2word, model.syn0)}






class TfidfEmbeddingVectorizer(object):
	def __init__(self, word2vec):
		self.word2vec = word2vec
		self.word2weight = None
		self.dim = len(word2vec.itervalues().next())
		
	def fit(self, X, y):
		tfidf = TfidfVectorizer(analyzer=lambda x: x)
		tfidf.fit(X)
		# if a word was never seen - it must be at least as infrequent
		# as any of the known words - so the default idf is the max of 
		# known idf's
		max_idf = max(tfidf.idf_)
		self.word2weight = defaultdict(
			lambda: max_idf, 
			[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
	
		return self
	
	def transform(self, X):
		return np.array([
				np.mean([self.word2vec[w] * self.word2weight[w]
						 for w in words if w in self.word2vec] or
						[np.zeros(self.dim)], axis=0)
				for words in X
			])
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())
    
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

etree_glove_small_tfidf = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)), 
						("extra trees", ExtraTreesClassifier(n_estimators=200))])


etree_glove_small_tfidf.fit(X, y)



file_Name = "intent_glove.pkl"
fileObject = open(file_Name,'wb') 
joblib.dump(etree_glove_small_tfidf, fileObject)
fileObject.close()









while 1:
	X_test = []
	X_test.append(nltk.word_tokenize(raw_input('Enter: ')))
	x_test = np.array(X_test)
	print etree_glove_small_tfidf.predict(x_test)
	try:
		print etree_glove_small_tfidf.predict_proba(x_test)
	except:
		continue













































