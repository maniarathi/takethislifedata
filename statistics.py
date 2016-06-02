import operator
from nltk.stem import *
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import numpy as np

def general_stats():
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
	print "Size of vocab: %d" % len(vocab)
	#for word in vocab:
	#	print word[0] + '\t' + str(word[1])
	f = open('featured_vocab.txt', 'w')
	for word in vocab:
		if word[1] > 100 and word[1] < 10000:
			f.write(word[0] + '\n')
	f.close()

def num_posts():
	counts = dict()
	for i in range(5):
		counts[i] = 0
	f = open('final_rated_posts.csv', 'r')
	num = 0
	for line in f:
		num += 1
		attributes = line.split('@@')
		rating = int(attributes[2])
		if rating in (1, 2, 3, 4, 5):
			counts[rating-1] += 1
	f.close()
	for count in counts:
		print "%d\t%d" % (count+1, counts[count])
	print "Total: %d" % num
	return counts

def get_most_common_words_per_class():
	counts = num_posts()
	f = open('final_rated_posts.csv', 'r')
	counters = []
	for _ in range(5):
		counters.append(Counter())
	for line in f:
		attributes = line.split('@@')
		words = set(attributes[1].strip().split())
		label = int(attributes[2])
		if label in (1, 2, 3, 4, 5):
			counters[label-1] += Counter(list(words))
	for i in xrange(len(counters)):
		counter = counters[i]
		words = counter.keys()
		for word in words:
			counter[word] = float(counter[word])/float(counts[i])
			if counter[word] > 0.50 or counter[word] < 0.20:
				del counter[word]
	for counter in counters:
		print counter.most_common(10)

def get_similarity_of_words():
	f = open('final_rated_posts.csv', 'r')
	counters = []
	vocab_map = dict()
	index = 0
	for _ in range(5):
		counters.append(Counter())
	for line in f:
		attributes = line.split('@@')
		words = set(attributes[1].strip().split())
		words_mapped = []
		for word in words:
			if word not in vocab_map:
				vocab_map[word] = index
				index += 1
			words_mapped.append(vocab_map[word])
		label = int(attributes[2])
		if label in (1, 2, 3, 4, 5):
			counters[label-1] += Counter(list(words_mapped))
	f.close()
	vectorizer = DictVectorizer(sparse=False)
	data = vectorizer.fit_transform(counters)
	words_1 = set(counters[0].keys())
	words_5 = set(counters[4].keys())
	print len(words_1)
	print len(words_5)
	print len(words_1 - words_5)
	print len(words_5 - words_1)
	for i in xrange(len(data)):
		words_i = data[i]
		for j in xrange(len(data)):
			words_j = data[j]
			print "Comparing words of posts categorized as %d and %d" % (i + 1, j + 1)
			#print cosine_similarity(words_i, words_j)
			print jaccard_sim(set(counters[i]), set(counters[j]))

def plot_confusion_matrix(title='Cosine Similarity of Label Vocabulary', cmap=plt.cm.Blues):
	cosine_similarity = [[1, 0.99080533, 0.99142912, 0.99054302, 0.98702335], [0.99080533, 1, 0.99552185, 0.99492576, 0.99100906], [0.99142912, 0.99552185, 1, 0.99574034, 0.99216503], [0.99054302, 0.99492576, 0.99574034, 1, 0.99172614], [0.98702335, 0.99100906, 0.99216503, 0.99172614, 1]]
	plt.imshow(cosine_similarity, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	target_names = [1,2,3,4,5]
	tick_marks = np.arange(len(target_names))
	plt.xticks(tick_marks, target_names, rotation=0)
	plt.yticks(tick_marks, target_names)
	plt.tight_layout()
	plt.ylabel('')
	plt.xlabel('')

if __name__ == "__main__":
	#general_stats()
	#num_posts()
	#get_most_common_words_per_class()
	#get_similarity_of_words()
	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix()
	plt.savefig("confusion_matrix.png")
	plt.close()