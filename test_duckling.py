from duckling import DucklingWrapper
import json
d = DucklingWrapper()

while 1:
	try:
		sentence = raw_input("Enter: ")

		dic = {'out_put': d.parse_time(sentence)}
		print d.parse_time(sentence)
		file_open = open('duckling_time_new.json','w')
		json.dump(dic,file_open)
		file_open.close()
	except Exception,e:
		print e

