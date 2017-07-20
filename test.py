import json
import re
from datetime import datetime
while 1:
	message = raw_input('Enter: ')
	probable_dates = []
	for each_date in re.finditer(r'((\d+/\d+/\d+)|(\d+-\d+-\d+))', message):
		try:
			try:
				probable_dates.append(datetime.strptime(each_date.group(1), "%d/%m/%Y"))
			except:
				probable_dates.append(datetime.strptime(each_date.group(1), "%d-%m-%Y"))
		except:
			print 'Please restrict your self to date month and year format only.'
	print probable_dates


# import datetime
# import re

# def valid_date(datestring):
#         try:
#                 mat=re.match('(\d{2})[/.-](\d{2})[/.-](\d{4})$', datestring)
#                 if mat is not None:
#                         datetime.datetime(*(map(int, mat.groups()[-1::-1])))
#                         return True
#         except ValueError:
#                 pass
#         return False
	# for each_day in re.finditer(r'(\d\s|\d\w+\s)', message):
	# 	print each_day.group(1)

# a = []

# def test():
# 	global a
# 	a.append(1)
# 	a.append(2)
# 	print a[-1]
# 	print a.pop()
# 	print a
# def test_1():
# 	global a
# 	a.append(3)
# if __name__ == '__main__':
# 	test_1()
# 	test()
	
