from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

lmtzr = WordNetLemmatizer()
import json



file_open = open('data_1.json','r')
data = json.load(file_open)
file_open.close()

vectorizer = TfidfVectorizer(analyzer='word',lowercase=True,use_idf=True,ngram_range=(1,3))

training_text = []
training_class = []

#Stopwords generation
stop_words = []
file_open_stopwords = open('stop_words.txt','r')
for each_word in file_open_stopwords:
	stop_words.append(each_word.strip())
file_open_stopwords.close()

for label in data.keys():
	for text in data[label]["examples"]:
		training_class.append(label)
		text_clean = [each_word for each_word in text.split() if each_word not in stop_words] 
		text_lmtzr = [lmtzr.lemmatize(each_word) for each_word in text_clean]
		training_text.append(' '.join(text_lmtzr))



X_vector = vectorizer.fit_transform(training_text)

print "  Actual number of tfidf features: %d" % X_vector.get_shape()[1]
# raw_input()

svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

X_train_lsa = lsa.fit_transform(X_vector)

passive_tfidf = PassiveAggressiveClassifier(n_iter=50)
passive_tfidf.fit(X_vector, training_class)

passive_lsa = PassiveAggressiveClassifier(n_iter=50)
passive_lsa.fit(X_train_lsa, training_class)

file_Name = "intent_leave_tfidf.p"
fileObject = open(file_Name,'wb') 
pickle.dump(passive_tfidf, fileObject)
fileObject.close()

file_Name_lsa = "intent_leave_lsa.p"
fileObject_lsa = open(file_Name_lsa,'wb') 
pickle.dump(passive_lsa, fileObject_lsa)
fileObject_lsa.close()


while(1):
	out_put = []
	test_text = raw_input('Enter: ')
	test_text_clean = [each_word for each_word in test_text.split() if each_word not in stop_words] 
	test_text_lmtzr = [lmtzr.lemmatize(each_word) for each_word in test_text_clean]
	out_put.append(' '.join(test_text_lmtzr))
	out_put_vector = vectorizer.transform(out_put)
	out_put_class = passive_tfidf.predict(out_put_vector)
	print 'tf-idf: ',out_put_class
	print 'tf-idf: ',passive_tfidf.decision_function(out_put_vector)
	out_put_vector_lsa = lsa.transform(out_put_vector)
	print 'lsa: ',passive_lsa.predict(out_put_vector_lsa)
	print 'lsa: ',passive_lsa.decision_function(out_put_vector_lsa)