import pickle
from nltk.stem.wordnet import WordNetLemmatizer 
lmtzr = WordNetLemmatizer()


#Stopwords generation
stop_words = []
file_open_stopwords = open('stop_words.txt','r')
for each_word in file_open_stopwords:
	stop_words.append(each_word.strip())
file_open_stopwords.close()



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