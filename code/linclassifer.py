glove_home = 'glove.6B'

import os
from collections import Counter, defaultdict
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score, f1_score
import utils
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer

GLOVE = utils.glove2dict(os.path.join(glove_home, 'glove.6B.100d.txt'))

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
        return 1
    elif y in (4, 5):
        return 2
    else:
        return None
    
def ternary_class_func(y):       
    if y in (0, 1):
        return 1
    elif y in (3, 4):
        return 3
    else:
        return 2

def original_class(y):
    return y

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
        all_feats.append(Counter(text))
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

def build_dataset(phi, class_func):
    responses, labels = get_labeled_data(class_func=class_func)
    data = phi(responses)
    return data, labels

def run_experiment(model, phi=glove_phi, class_func=original_class):
    X, y = build_dataset(phi, class_func)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    score, predicted_labels = model(X_train, X_test, y_train, y_test)
    print "Ran %s" % model.__name__
    print "Accuracy on test set: %f" % score
    print "F1 Score of test set: " 
    print f1_score(y_test, predicted_labels, average='micro')

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


###############################################################################
# END MODELS HERE
###############################################################################

if __name__ == "__main__":
    #run_experiment(k_means)
    #run_experiment(SVC)
    #run_experiment(SGD)
    #run_experiment(k_means, phi=unigram_phi)
    #run_experiment(SVC, phi=unigram_phi)
    #run_experiment(SGD, phi=unigram_phi)
    run_experiment(k_means, phi=bigram_phi)
    run_experiment(SVC, phi=bigram_phi)
    run_experiment(SGD, phi=bigram_phi)
