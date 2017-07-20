# import datefinder
# while 1:
# 	sentence = raw_input("Enter: ")
# 	matches = datefinder.find_dates(sentence)
# 	matches_list = [match for match in matches]
# 	print len(matches_list)
# 	for match in matches_list:
# 		print match
# 		print type(match)

from datetime import datetime
from natty import DateParser
while 1:
	sentence = raw_input('Enter: ')
	dp = DateParser(sentence)
	print dp.result()[0].date().strftime('%d-%m-%Y')
	print type(dp.result()[0].date().strftime('%d-%m-%Y'))

# import datetime
# from recurrent import RecurringEvent
# while  1:
# 	r = RecurringEvent(now_date=datetime.datetime(2017, 7, 14))
# 	sentence = raw_input('Enter: ')
# 	print r.parse(sentence)
# 	print r.get_params()

# from dateutil.parser import parse

# def is_date(string):
#     try: 
#         parse(string)
#         return True
#     except ValueError:
#         return False
# if __name__ == '__main__':
# 	while 1:
# 		print is_date(raw_input('Enter: '))