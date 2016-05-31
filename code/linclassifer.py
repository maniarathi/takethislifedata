glove_home = 'glove.6B'

import os
import unicodedata
import utils
from nltk.stem import *
from collections import Counter, defaultdict
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment

GLOVE = utils.glove2dict(os.path.join(glove_home, 'glove.6B.100d.txt'))
featured_vocab_file = '../featured_vocab.txt'

def plotTSNE(data):
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    tsne_data = model.fit_transform(data)

    x1, x2 = zip(*tsne_data)

    # print knn_score, svm_score

    import matplotlib.pyplot as plt
    plt.scatter(x1, x2, c=labels)
    plt.show()

    # f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    # ax2.scatter(x1, x2, c=knn_reduced_labels)
    # ax2.set_title("KNN clusters")
    # ax1.scatter(x1, x2, c=test_labels)
    # ax1.set_title("Labels")
    # ax3.scatter(x1, x2, c=svm_labels)
    # ax3.set_title("SVM classification")
    # plt.show()

def binary_class_func(y):
    if y in (1, 2):
        return 0
    elif y in (4, 5):
        return 1
    else:
        return None

def ternary_class_func(y):
    if y in (1, 2):
        return 0
    elif y in (4, 5):
        return 2
    elif y in (3):
        return 1
    else:
        return None

def original_class(y):
    if y in (1, 2, 3, 4, 5):
        return y-1
    else:
        return None

###############################################################################
# BEGIN FEATURE EXTRACTORS HERE
###############################################################################
def glove_phi(texts):
    unknown = defaultdict(int)
    data = np.zeros((len(texts), 100))
    for i, r in enumerate(texts):
        words = r.split(" ")
        vec = []
        for word in words:
            if word in GLOVE:
                vec.append(GLOVE[word])
            else:
                unknown[word] += 1
        vec = np.array(vec)
        curSum = np.sum(vec, axis=0)
        data[i] = curSum
    data = PCA(n_components='mle').fit_transform(data)
    return data

def unigram_phi(texts):
    all_feats = []
    for text in texts:
        all_feats.append(Counter(text.split(" ")))
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(all_feats)

def stemmed_unigram_phi(texts):
    all_feats = []
    stemmer = PorterStemmer()
    for text in texts:
        words = text.split(" ")
        clean_words = []
        for word in words:
            word = unicodedata.normalize('NFKD',unicode(word,"ISO-8859-1")).encode("ascii","ignore")
            clean_words.append(word.lower())
        stemmed_words = [stemmer.stem(word) for word in clean_words]
        all_feats.append(Counter(stemmed_words))
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(all_feats)

def stemmed_featured_unigrams_phi(texts):
    all_feats = []
    stemmer = PorterStemmer()
    f = open(featured_vocab_file, 'r')
    stemmed_featured_words = set([])
    for line in f:
        word = line.strip()
        word = unicodedata.normalize('NFKD',unicode(word,"ISO-8859-1")).encode("ascii","ignore")
        stemmed_featured_words.add(stemmer.stem(word))
    for text in texts:
        words = text.split(" ")
        clean_words = []
        for word in words:
            word = unicodedata.normalize('NFKD',unicode(word,"ISO-8859-1")).encode("ascii","ignore")
            clean_words.append(word.lower())
        stemmed_words = [stemmer.stem(word) for word in clean_words if stemmer.stem(word) in stemmed_featured_words]
        all_feats.append(Counter(stemmed_words))
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(all_feats)

def stemmed_featured_unigrams_phi_plus_sentiment(texts):
    all_feats = []
    stemmer = PorterStemmer()
    f = open(featured_vocab_file, 'r')
    stemmed_featured_words = set([])
    for line in f:
        word = line.strip()
        word = unicodedata.normalize('NFKD',unicode(word,"ISO-8859-1")).encode("ascii","ignore")
        stemmed_featured_words.add(stemmer.stem(word))
    for text in texts:
        words = text.split(" ")
        clean_words = []
        for word in words:
            word = unicodedata.normalize('NFKD',unicode(word,"ISO-8859-1")).encode("ascii","ignore")
            clean_words.append(word.lower())
        stemmed_words = [stemmer.stem(word) for word in clean_words if stemmer.stem(word) in stemmed_featured_words]
        text_features = Counter(stemmed_words)
        vs = vaderSentiment(text)
        text_features['neg_sentiment'] = vs['neg']
        text_features['neu_sentiment'] = vs['neu']
        text_features['pos_sentiment'] = vs['pos']
        all_feats.append(text_features)
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(all_feats)

def bigram_phi(texts):
    all_feats = []
    for text in texts:
        words = text.split(" ")
        bigrams = []
        for i in xrange(len(words) - 1):
            bigrams.append(words[i] + " " + words[i+1])
        features = Counter(bigrams)
        all_feats.append(features)
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(all_feats)

###############################################################################
# END FEATURE EXTRACTORS HERE
###############################################################################

def get_labeled_data(filename='../final_rated_posts.csv', class_func=original_class):
    with open(filename, 'rb') as inpFile:
        responses = []
        labels = []
        for line in inpFile:
            fields = line.split("@@")
            if len(fields) != 3:
                continue
            text = fields[1]
            label = class_func(int(fields[2]))
            if label != None:
                responses.append(text)
                labels.append(label)
    print "Size of all labeled data (train + test) = %d" % len(responses)
    return responses, labels

def build_dataset(posts, phi, class_func):
    data = phi(posts)
    return data

def run_experiment(posts, labels, model, phi=glove_phi, class_func=original_class):
    X = build_dataset(posts, phi, class_func)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500, random_state=42)
    score, predicted_labels = model(X_train, X_test, y_train, y_test)
    print "Ran %s" % model.__name__
    #print "Accuracy on test set: %f" % score
    #print "F1 Score of test set: "
    #print f1_score(y_test, predicted_labels, average='micro')
    print classification_report(y_test, predicted_labels)

###############################################################################
# BEGIN MODELS HERE
###############################################################################
def k_means(X_train, X_test, y_train, y_test):
    mod = KMeans(init='k-means++', n_clusters=5, n_init=10)
    mod.fit(X_train)
    knn_labels = mod.predict(X_test)
    knn_score = 0
    for i, l in enumerate(knn_labels):
        if l == y_test[i]:
            knn_score += 1
    knn_score = float(knn_score)/len(y_test)
    return knn_score, knn_labels

def nearest_neighbors(X_train, X_test, y_train, y_test):
    neigh = NearestNeighbors()
    neigh.fit(X_train, y_train)

def SVC(X_train, X_test, y_train, y_test):
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    svm_labels = clf.predict(X_test)
    svm_score = clf.score(X_test, y_test)
    return svm_score, svm_labels

def SGD(X_train, X_test, y_train, y_test):
    mod = SGDClassifier(fit_intercept=True)
    mod.fit(X_train, y_train)
    sgd_labels = mod.predict(X_test)
    sgd_score = mod.score(X_test, y_test)
    return sgd_score, sgd_labels

def NaiveBayes(X_train, X_test, y_train, y_test):
    mod = MultinomialNB()
    mod.fit(X_train, y_train)
    nb_labels = mod.predict(X_test)
    nb_score = mod.score(X_test, y_test)
    return nb_score, nb_labels

###############################################################################
# END MODELS HERE
###############################################################################

if __name__ == "__main__":
    posts, labels = get_labeled_data(class_func=original_class)
    #run_experiment(posts, labels, k_means)
    #run_experiment(posts, labels, SVC)
    #run_experiment(posts, labels, SGD)
    #run_experiment(posts, labels, NaiveBayes)
    #run_experiment(posts, labels, k_means, phi=unigram_phi)
    #run_experiment(posts, labels, SVC, phi=unigram_phi)
    #run_experiment(posts, labels, SGD, phi=unigram_phi)
    #run_experiment(posts, labels, NaiveBayes, phi=unigram_phi)
    #run_experiment(posts, labels, k_means, phi=stemmed_unigram_phi)
    #run_experiment(posts, labels, SVC, phi=stemmed_unigram_phi)
    #run_experiment(posts, labels, SGD, phi=stemmed_unigram_phi)
    #run_experiment(posts, labels, NaiveBayes, phi=stemmed_unigram_phi)
    #run_experiment(posts, labels, k_means, phi=stemmed_featured_unigrams_phi)
    #run_experiment(posts, labels, SVC, phi=stemmed_featured_unigrams_phi)
    #run_experiment(posts, labels, SGD, phi=stemmed_featured_unigrams_phi)
    #run_experiment(posts, labels, NaiveBayes, phi=stemmed_featured_unigrams_phi)
    run_experiment(posts, labels, k_means, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    run_experiment(posts, labels, SVC, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    run_experiment(posts, labels, SGD, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    run_experiment(posts, labels, NaiveBayes, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    #run_experiment(posts, labels, k_means, phi=bigram_phi)
    #run_experiment(posts, labels, SVC, phi=bigram_phi)
    #run_experiment(posts, labels, SGD, phi=bigram_phi)
    #run_experiment(posts, labels, NaiveBayes, phi=bigram_phi)
