import operator
from nltk.stem import *

f = open('final_rated_posts.csv', 'r')
vocab = dict()
num = 0
for line in f:
	num += 1
	attributes = line.split('@@')
	words = attributes[1].strip().split()
	for word in words:
		word = word.lower()
		if word in vocab:
			vocab[word] += 1
		else:
			vocab[word] = 1
f.close()
print "Number of lines: %d" % num
vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
for word in vocab:
	print word[0] + '\t' + str(word[1])
f = open('featured_vocab.txt', 'w')
for word in vocab:
	if word[1] > 100 and word[1] < 10000:
		f.write(word[0] + '\n')
f.close()
