# import nltk
# from nltk.corpus import stopwords
# stop = set(stopwords.words('english'))
# file_open = open('stop_words.txt','w')

# for each_word in stop:
# 	file_open.write(each_word+'\n')
# file_open.close()



file_open = open('stop_words.txt','r')
for each_word in file_open:
	print each_word
	raw_input()

