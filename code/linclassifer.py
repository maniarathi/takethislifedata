glove_home = 'glove.6B'

import os
import unicodedata
import random
import utils
from nltk.stem import *
from collections import Counter, defaultdict
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import BernoulliRBM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
import matplotlib.pyplot as plt
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor

GLOVE = utils.glove2dict(os.path.join(glove_home, 'glove.6B.100d.txt'))
featured_vocab_file = '../featured_vocab.txt'

appreciation = ["thank you", "thanks", "appreciate"]
seeking_help = ["need help", "help me", "please help"]
finality = ["last", "forever"]
profanity = ["fuck", "asshole", "assholes", "f*****", "f****n", "bitch", "f*****g", "bloody", "goddamn", "motherfucking", "motherfuckers", "motherfucker", "shit", "cunt", "ass", 
"shitty", "fucking", "fucker", "fuckers"]
fuck_words = ["fuck", "f*****", "f****n", "fucker", "fuckers"]

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
    if y in (1, 2, 3):
        return 0
    elif y in (4, 5):
        return 1
    else:
        return None

def ternary_class_func(y):
    if y == 1:
        return 0
    elif y == 5:
        return 2
    elif y in (2, 3, 4):
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

def bigrams_unigrams_sentiment(texts):
    all_feats = []
    for text in texts:
        profanity_count = 0
        appreciation_count = 0
        help_count = 0
        for string in appreciation:
            appreciation_count += text.lower().count(string)
        for string in profanity:
            profanity_count += text.lower().count(string)
        for string in seeking_help:
            help_count += text.lower().count(string)
        words = text.split(" ")
        bigrams = []
        for i in xrange(len(words) - 1):
            bigrams.append(words[i] + " " + words[i+1])
        features = Counter(bigrams)
        features += Counter(words)
        vs = vaderSentiment(text)
        #print text, vs
        features['neg_sentiment'] = vs['neg']
        features['neu_sentiment'] = vs['neu']
        features['pos_sentiment'] = vs['pos']
        features['profanity_count'] = profanity_count
        features['appreciation_count'] = appreciation_count
        features['help_count'] = help_count
        features['text_length'] = len(text)
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

def build_dataset(posts, phi):
    data = phi(posts)
    return data

def run_experiment(posts, labels, model, phi=glove_phi, class_func=original_class):
    X = build_dataset(posts, phi)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500, random_state=42)
    score, predicted_labels = model(X_train, X_test, y_train, y_test)
    print "Ran %s using phi %s and class function %s" % (model.__name__, phi.__name__, class_func.__name__)
    #print "Accuracy on test set: %f" % score
    #print "F1 Score of test set: "
    #print f1_score(y_test, predicted_labels, average='micro')
    print classification_report(y_test, predicted_labels)
    #cm = confusion_matrix(y_test, predicted_labels)
    #np.set_printoptions(precision=2)
    #print 'Confusion matrix, without normalization'
    #print cm
    #plt.figure()
    #plot_confusion_matrix(cm)

def get_best_params(posts, labels, phi=glove_phi, class_func=original_class):
    X = build_dataset(posts, phi, class_func)
    y = labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=500, random_state=42)
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['precision', 'recall']

    for score in scores:
        print "# Tuning hyper-parameters for %s" % score
        clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,scoring='%s_weighted' % score)
        clf.fit(X_train, y_train)
        print "Best parameters set found on development set:"
        print clf.best_params_
        print "Grid scores on development set:"
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params)

        print "Detailed classification report:"
        print "The model is trained on the full development set."
        print "The scores are computed on the full evaluation set."
        y_true, y_pred = y_test, clf.predict(X_test)
        print classification_report(y_true, y_pred)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

###############################################################################
# BEGIN MODELS HERE
###############################################################################
def k_means(X_train, X_test, y_train, y_test):
    mod = KMeans(init='k-means++', n_clusters=5, n_init=10)
    mod.fit(X_train)
    print "Done training"
    knn_labels = mod.predict(X_test)
    print "Done testing"
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
    print "Done training"
    svm_labels = clf.predict(X_test)
    print "Done testing"
    svm_score = clf.score(X_test, y_test)
    return svm_score, svm_labels

def SGD(X_train, X_test, y_train, y_test):
    mod = SGDClassifier(fit_intercept=True, )
    mod.fit(X_train, y_train)
    print "Done training"
    sgd_labels = mod.predict(X_test)
    print "Done testing"
    sgd_score = mod.score(X_test, y_test)
    return sgd_score, sgd_labels

def NaiveBayes(X_train, X_test, y_train, y_test):
    mod = MultinomialNB()
    mod.fit(X_train, y_train)
    print "Done training"
    nb_labels = mod.predict(X_test)
    print "Done testing"
    nb_score = mod.score(X_test, y_test)
    return nb_score, nb_labels

def LogisticRegressor(X_train, X_test, y_train, y_test):
    mod = LogisticRegression()
    mod.fit(X_train, y_train)
    print "Done training"
    lr_labels = mod.predict(X_test)
    print "Done testing"
    lr_score = mod.score(X_test, y_test)
    return lr_score, lr_labels

def Bernoulli(X_train, X_test, y_train, y_test):
    mod = BernoulliRBM(random_state=0, verbose=True)
    mod.fit(X_train, y_train)
    print "Done training"
    bernoulli_labels = mod.predict(X_test)
    print "Done testing"
    bernoulli_score = mod.score(X_test, y_test)
    return bernoulli_score, bernoulli_labels

def LogisticRegressionWithSGD(X_train, X_test, y_train, y_test):
    mod = SGDClassifier(fit_intercept=False, loss="log", n_jobs=-1, random_state=42)
    mod.fit(X_train, y_train)
    print "Done training"
    lr_labels = mod.predict(X_test)
    print "Done testing"
    lr_score = mod.score(X_test, y_test)
    return lr_score, lr_labels

def GradientBoosted(X_train, X_test, y_train, y_test):
    mod = GradientBoostingRegressor()
    mod.fit(X_train, y_train)
    print "Done training"
    gb_labels = mod.predict(X_test)
    print "Done testing"
    gb_score = mod.score(X_test, y_test)
    return gb_score, gb_labels

###############################################################################
# END MODELS HERE
###############################################################################

if __name__ == "__main__":
    posts, labels = get_labeled_data(class_func=ternary_class_func)
    
    #run_experiment(posts, labels, k_means)
    #run_experiment(posts, labels, SVC)
    #run_experiment(posts, labels, SGD)
    
    #run_experiment(posts, labels, k_means, phi=glove_phi)
    #run_experiment(posts, labels, SVC, phi=glove_phi)
    #run_experiment(posts, labels, SGD, phi=glove_phi)
    #run_experiment(posts, labels, NaiveBayes, phi=glove_phi)
    #run_experiment(posts, labels, Bernoulli, phi=glove_phi)
    #run_experiment(posts, labels, LogisticRegressor, phi=glove_phi)
    #run_experiment(posts, labels, LogisticRegressionWithSGD, phi=glove_phi)
    #run_experiment(posts, labels, GradientBoosted, phi=glove_phi)

    #run_experiment(posts, labels, k_means, phi=unigram_phi)
    #run_experiment(posts, labels, SVC, phi=unigram_phi)
    #run_experiment(posts, labels, SGD, phi=unigram_phi)
    #run_experiment(posts, labels, NaiveBayes, phi=unigram_phi)
    #run_experiment(posts, labels, Bernoulli, phi=unigram_phi)
    #run_experiment(posts, labels, LogisticRegressor, phi=unigram_phi)
    
    #run_experiment(posts, labels, k_means, phi=stemmed_unigram_phi)
    #run_experiment(posts, labels, SVC, phi=stemmed_unigram_phi)
    #run_experiment(posts, labels, SGD, phi=stemmed_unigram_phi)
    #run_experiment(posts, labels, NaiveBayes, phi=stemmed_unigram_phi)
    #run_experiment(posts, labels, Bernoulli, phi=stemmed_unigram_phi)
    #run_experiment(posts, labels, LogisticRegressor, phi=stemmed_unigram_phi)
    
    #run_experiment(posts, labels, k_means, phi=stemmed_featured_unigrams_phi)
    #run_experiment(posts, labels, SVC, phi=stemmed_featured_unigrams_phi)
    #run_experiment(posts, labels, SGD, phi=stemmed_featured_unigrams_phi)
    #run_experiment(posts, labels, NaiveBayes, phi=stemmed_featured_unigrams_phi)
    #run_experiment(posts, labels, Bernoulli, phi=stemmed_featured_unigrams_phi)
    #run_experiment(posts, labels, LogisticRegressor, phi=stemmed_featured_unigrams_phi)
    
    #run_experiment(posts, labels, k_means, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    #run_experiment(posts, labels, SVC, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    #run_experiment(posts, labels, SGD, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    #run_experiment(posts, labels, NaiveBayes, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    #run_experiment(posts, labels, Bernoulli, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    #run_experiment(posts, labels, LogisticRegressor, phi=stemmed_featured_unigrams_phi_plus_sentiment)
    
    #run_experiment(posts, labels, k_means, phi=bigram_phi)
    #run_experiment(posts, labels, SVC, phi=bigram_phi)
    #run_experiment(posts, labels, SGD, phi=bigram_phi)
    #run_experiment(posts, labels, NaiveBayes, phi=bigram_phi)
    #run_experiment(posts, labels, Bernoulli, phi=bigram_phi)
    #run_experiment(posts, labels, LogisticRegressor, phi=bigram_phi)

    run_experiment(posts, labels, k_means, phi=bigrams_unigrams_sentiment)
    run_experiment(posts, labels, SVC, phi=bigrams_unigrams_sentiment)
    run_experiment(posts, labels, SGD, phi=bigrams_unigrams_sentiment)
    run_experiment(posts, labels, NaiveBayes, phi=bigrams_unigrams_sentiment)
    #run_experiment(posts, labels, Bernoulli, phi=bigrams_unigrams_sentiment)
    run_experiment(posts, labels, LogisticRegressor, phi=bigrams_unigrams_sentiment)
    run_experiment(posts, labels, LogisticRegressionWithSGD, phi=bigrams_unigrams_sentiment)
    #run_experiment(posts, labels, GradientBoosted, phi=bigrams_unigrams_sentiment)
    
    #get_best_params(posts, labels, phi=glove_phi, class_func=original_class)
    #get_best_params(posts, labels, phi=unigram_phi, class_func=original_class)
    #get_best_params(posts, labels, phi=stemmed_unigram_phi, class_func=original_class)
    #get_best_params(posts, labels, phi=stemmed_featured_unigrams_phi, class_func=original_class)
    #get_best_params(posts, labels, phi=stemmed_featured_unigrams_phi_plus_sentiment, class_func=original_class)
    #get_best_params(posts, labels, phi=bigram_phi, class_func=original_class)

    #get_best_params(posts, labels, phi=glove_phi, class_func=binary_class_func)
    #get_best_params(posts, labels, phi=unigram_phi, class_func=binary_class_func)
    #get_best_params(posts, labels, phi=stemmed_unigram_phi, class_func=binary_class_func)
    #get_best_params(posts, labels, phi=stemmed_featured_unigrams_phi, class_func=binary_class_func)
    #get_best_params(posts, labels, phi=stemmed_featured_unigrams_phi_plus_sentiment, class_func=binary_class_func)
    #get_best_params(posts, labels, phi=bigram_phi, class_func=binary_class_func)
