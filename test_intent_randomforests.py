from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
import pickle

lmtzr = WordNetLemmatizer()
import json



file_open = open('data_global_intent.json','r')
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

random_forest_tfidf = RandomForestClassifier(n_estimators=100,max_depth=None,max_features='sqrt',min_samples_split=2,bootstrap=True,oob_score=True)
random_forest_tfidf.fit(X_vector, training_class)


random_forest_lsa = RandomForestClassifier(n_estimators=100,max_depth=None,max_features='sqrt',min_samples_split=2,bootstrap=True,oob_score=True)
random_forest_lsa.fit(X_train_lsa, training_class)


file_Name = "intent_leave_tfidf_random_forests.p"
fileObject = open(file_Name,'wb') 
pickle.dump(random_forest_tfidf, fileObject)
fileObject.close()

file_Name_lsa = "intent_leave_lsa_random_forests.p"
fileObject_lsa = open(file_Name_lsa,'wb') 
pickle.dump(random_forest_lsa, fileObject_lsa)
fileObject_lsa.close()

# file_vectorizer_open = open("tfidf_vectorizer.p",'wb')
# pickle.dump(vectorizer,file_vectorizer_open)
# file_vectorizer_open.close()

# file_lsa_vectorizer_open = open("lsa_vectorizer.p",'wb')
# pickle.dump(lsa,file_lsa_vectorizer_open)
# file_lsa_vectorizer_open.close()

while(1):
	out_put = []
	test_text = raw_input('Enter: ')
	test_text_clean = [each_word for each_word in test_text.split() if each_word not in stop_words] 
	test_text_lmtzr = [lmtzr.lemmatize(each_word) for each_word in test_text_clean]
	out_put.append(' '.join(test_text_lmtzr))
	out_put_vector = vectorizer.transform(out_put)
	out_put_class = random_forest_tfidf.predict(out_put_vector)
	print 'tf-idf: ',out_put_class
	print 'tf-idf: ',random_forest_tfidf.predict_proba(out_put_vector)
	out_put_vector_lsa = lsa.transform(out_put_vector)
	print 'lsa: ',random_forest_lsa.predict(out_put_vector_lsa)
	print 'lsa: ',random_forest_lsa.predict_proba(out_put_vector_lsa)
	