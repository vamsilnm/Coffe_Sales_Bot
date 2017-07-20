import spacy
nlp = spacy.load('en')
size_list = []
example = (u'The cat is chasing the dog.')
parse = nlp(example)

# print parse
for word in parse:
	print word
	size_list.append(word.dep_)
	# if word.dep_ == 'nsubj':
	#     print('Subject:', word.text)
	# if word.dep_ == 'dobj':
	#     print('Object:', word.text)


print (size_list)