import numpy as np
#convert from binary to string, x is numpy matrix (#chars, 11)
char_dict = {
	'1,0,0,0,0,0,0,0,0,0,0':'e',
	'0,1,0,0,0,0,0,0,0,0,0':'t',
	'0,0,1,0,0,0,0,0,0,0,0':'a',
	'0,0,0,1,0,0,0,0,0,0,0':'o',
	'0,0,0,0,1,0,0,0,0,0,0':'i',
	'0,0,0,0,0,1,0,0,0,0,0':'n',
	'0,0,0,0,0,0,1,0,0,0,0':'s',
	'0,0,0,0,0,0,0,1,0,0,0':'h',
	'0,0,0,0,0,0,0,0,1,0,0':'r',
	'0,0,0,0,0,0,0,0,0,1,0':'d',
	'0,0,0,0,0,0,0,0,0,0,1':' ',
}
def decode(x):
	output = []
	for i in range(x.shape[0]):
		cur_char = np.array_str(x[i, :]).replace('.','').replace('  ', ',').replace('[', '').replace(']', '').replace(' ', '')
		char_f = char_dict.get(cur_char, '?')
		output.append(char_f)

	return output

def encode(x):
	char_dict_rev = dict ( (v,k) for k, v in char_dict.items() )
	output = np.zeros((len(x), 11))
	i = 0
	for char in x:
		cur_char = char_dict_rev.get(char, None)
		char_f = np.matrix(cur_char)
		output[i] = char_f
		i += 1
	return output