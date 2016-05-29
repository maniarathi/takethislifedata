import operator

f = open('posts.csv', 'r')
vocab = dict()
num = 0
for line in f:
	if len(line) <= 1:
		num += 1
	else:
		words = line.strip().split()
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