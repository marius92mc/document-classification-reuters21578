import time
import html
from pprint import pprint
import re
from HTMLParser import HTMLParser
import pickle

from reuters_parser import ReutersParser

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


def obtain_train_test_tags():
    types = open("data/all-cgisplit.txt", "r").readlines()
    types = [t.strip() for t in types]
    for i in range(0, len(types)):
        types[i] = types[i].lower()
    return types 


def get_most_important_topics():
    topics = open("data/most_popular_topics.txt", "r").readlines()
    topics = [t.strip() for t in topics]
    for i in range(0, len(topics)):
        topics[i] = topics[i].lower()
    return topics


def obtain_topic_tags():
    """
    Open the topic list file and import all of the topic names
    taking care to strip the trailing "\n" from each word.
    """
    topics = open("data/all-topics-strings.lc.txt", "r").readlines()
    topics = [t.strip() for t in topics]
    for i in range(0, len(topics)):
        topics[i] = topics[i].lower()
    return topics


def filter_doc_list_through_topics_train_test(topics, types, docs):
    """
    Reads all of the documents and creates a new list of two-tuples
    that contain a single feature entry and the body text, instead of
    a list of topics. It removes all geographic features and only 
    retains those documents which have at least one non-geographic
    topic.
    """
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        if d[2] not in types:
            continue
        label = d[0]
        if label in topics:
            d_tup = (label, d[1], d[2])
            ref_docs.append(d_tup)
    return ref_docs


def create_tfidf_data(docs, doc_type, k_train_tag, k_test_tag):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list. 

    The function returns both the class label vector (y) and 
    the corpus token/feature matrix (X).
    """
    docs = [doc for doc in docs if doc[2] == doc_type]
    # Create the training data class labels
    y = [d[0] for d in docs]
    
    # Create the document corpus list
    corpus = [d[1] for d in docs]
    #print corpus[0]
    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    if doc_type == k_train_tag:
        x = vectorizer.fit_transform(corpus)
    else:
        x = vectorizer.transform(corpus)
    return x, y, vectorizer


def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
    print "Training..."
    start_time = time.time() 
    svm.fit(X, y)
    print "Training time %.2f seconds." % (time.time() - start_time)
    return svm


def get_number_of_train_test_from_docs(ref_docs):
    count_train = 0
    count_test = 0
    for doc in ref_docs:
        if len(doc) > 2:
            if doc[2] == "training-set":
                count_train += 1
            if doc[2] == "published-testset":
                count_test += 1
    return count_train, count_test 


def get_count_docs_per_label_per_type(ref_docs, topics, doc_type):
    count_label0 = 0
    count_label1 = 0
    if len(topics) < 2:
        return 0 
    for doc in ref_docs:
        if len(doc) < 3: 
            continue
        if doc[2] != doc_type:
            continue
        if doc[0] == topics[0]:
            count_label0 += 1
        if doc[0] == topics[1]:
            count_label1 += 1
    return count_label0, count_label1


def print_test_summary(ref_docs, x_test, topics, \
                       labels, pred, svm, \
                       k_train_tag, k_test_tag):
    print len(ref_docs), "Docs"
    count_train_docs, count_test_docs = \
                    get_number_of_train_test_from_docs(ref_docs)
    print count_train_docs, "Train Docs"
    count_label0, count_label1 = \
                    get_count_docs_per_label_per_type(ref_docs, \
                                                      topics, \
                                                      k_train_tag)
    print "\t", count_label0, topics[0] 
    print "\t", count_label1, topics[1]

    print count_test_docs, "Test Docs"
    count_label0, count_label1 = \
                    get_count_docs_per_label_per_type(ref_docs, \
                                                      topics, \
                                                      k_test_tag)
    print "\t", count_label0, topics[0] 
    print "\t", count_label1, topics[1]
    """
    for i in range(0, len(pred)):
        print labels[i], pred[i] 
        if labels[i] == pred[i]:
            print "OK"
        else:
            print "NOK"
    """
    # Output the hit-rate and the confusion matrix for each model
    score = svm.score(x_test, labels)
    print "\nHit-rate score ", score
    """ By definition a confusion matrix C is such that C(i, j) is 
    equal to the number of observations known to be in group i, 
    but predicted to be in group j.
    """
    print "Confusion matrix"
    print confusion_matrix(pred, labels)
    metrics = precision_recall_fscore_support(labels, pred, \
                                              pos_label=2)
    if len(metrics) != 4:
        return score 
    print "Precision %.3f" % metrics[0][0]
    print "Recall %.3f" % metrics[1][0]
    print "fbeta_score %.3f" % metrics[2][0]
    #print "Support", metrics[3][0]
    #print classification_report(labels, pred)
    return score, metrics[0][0], metrics[1][0], metrics[2][0]
    

def test_and_print(x_train, topics, k_train_tag, \
                   k_test_tag, ref_docs, vectorizer, svm):
    start_time = time.time()
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train)

    corpus_test = [doc[1] for doc in ref_docs if doc[2] == k_test_tag]        
    labels = [doc[0] for doc in ref_docs if doc[2] == k_test_tag]
        
    x_test_counts = vectorizer.transform(corpus_test)
    x_test = tfidf_transformer.transform(x_test_counts) 
    
    # Make an array of predictions on the test set
    prediction = svm.predict(x_test)
    # print "\nHit-rate score is ", svm.score(X_test, labels)
    print "Testing time %.2f seconds." % (time.time() - start_time)
    return print_test_summary(ref_docs, x_test, topics, \
                              labels, prediction, svm, \
                              k_train_tag, k_test_tag)
    
def get_average(collection):
    sum = 0
    for entry in collection:
        sum += entry 
    return float(sum / len(collection))

def main():
    start_time = time.time()
    # Create the list of Reuters data and create the parser
    files = ["data/reut2-%03d.sgm" % r for r in range(0, 22)]
    parser = ReutersParser()
    k_train_tag = "training-set"
    k_test_tag = "published-testset" 

    # Parse the document and force all generated docs into
    # a list so that it can be printed out to the console
    print "Parsing training data...\n"
    docs = []
    for fn in files:
        for d in parser.parse(open(fn, 'rb')):
            docs.append(d)
    # Obtain the topic tags and filter docs through it 
    topics = get_most_important_topics()
    types = obtain_train_test_tags() 
    
    all_docs = docs 
    all_topics = topics 
    scores = []
    precisions = []
    recalls = []
    f1_values = []
    num_test = 0

    for topic in all_topics:
        # consider labels as topic and non-topic
        docs = []
        topics = [topic, "non-" + topic] 
        if len(topics) != 2:
            continue
        print "Test %d/%d" % ((num_test + 1), len(all_topics)) 
        print "----------", \
              "Labels ", topics[0], topics[1], \
              "----------"

        for doc in all_docs:
            if len(doc) < 3:
                continue 
            label = "non-" + topic 
            for topic_entry in doc[0]:
                if topic_entry == topic:
                    label = topic 
                    break
            docs.append((label, doc[1], doc[2]))
        
        ref_docs = filter_doc_list_through_topics_train_test(topics, \
                                                             types, \
                                                             docs)
        
        # Vectorise and TF-IDF transform the corpus 
        x_train, labels, vectorizer = create_tfidf_data(ref_docs, \
                                                        k_train_tag, \
                                                        k_train_tag, \
                                                        k_test_tag)
        # Create and train the Support Vector Machine
        svm = train_svm(x_train, labels)
        # Save data from svm
        #print "Saving data from training..."
        #joblib.dump(svm, "saved_data/saved_trained_data.pkl")
        #joblib.dump(vectorizer, "saved_data/saved_tfidfvectorizer_instance.pkl")

        score, precision, recall, f1_value = test_and_print(x_train, \
                                                            topics, \
                                                            k_train_tag, \
                                                            k_test_tag, \
                                                            ref_docs, \
                                                            vectorizer, \
                                                            svm)
        scores.append(score)
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1_value)
        print "\n\n"
        num_test += 1

    average_hit_score = get_average(scores)
    average_precision = get_average(precisions)
    average_recall = get_average(recalls) 
    average_f1 = get_average(f1_values)
    print "\n"
    print "Average Hit-rate score %.3f" % average_hit_score 
    print "Average Precision %.3f" % average_precision 
    print "Average Recall %.3f" % average_recall 
    print "Average f1 %.3f" % average_f1
    print "Total runtime %.2f seconds.\n" % (time.time() - start_time)
    

if __name__ == "__main__":
    main()
    
