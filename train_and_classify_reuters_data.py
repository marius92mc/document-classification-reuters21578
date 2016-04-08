import time
import html
from pprint import pprint
import re
from HTMLParser import HTMLParser
import pickle

from reuters_parser import ReutersParser

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.externals import joblib



def obtain_topic_tags():
    """
    Open the topic list file and import all of the topic names
    taking care to strip the trailing "\n" from each word.
    """
    topics = open("data/all-topics-strings.lc.txt", "r").readlines()
    topics = [t.strip() for t in topics]
    return topics

def filter_doc_list_through_topics(topics, docs):
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
        for t in d[0]:
            if t in topics:
                d_tup = (t, d[1])
                ref_docs.append(d_tup)
                break
    return ref_docs

def create_tfidf_training_data(docs):
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list. 

    The function returns both the class label vector (y) and 
    the corpus token/feature matrix (X).
    """
    # Create the training data class labels
    y = [d[0] for d in docs]
    
    # Create the document corpus list
    corpus = [d[1] for d in docs]
    #print corpus[0]
    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
    X = vectorizer.fit_transform(corpus)
    
    return X, y, vectorizer

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

def main():
    start_time = time.time()
    # Create the list of Reuters data and create the parser
    files = ["data/reut2-%03d.sgm" % r for r in range(0, 22)]
    parser = ReutersParser()

    # Parse the document and force all generated docs into
    # a list so that it can be printed out to the console
    print "Parsing training data..."
    docs = []
    for fn in files:
        for d in parser.parse(open(fn, 'rb')):
            docs.append(d)

    # Obtain the topic tags and filter docs through it 
    topics = obtain_topic_tags()
    ref_docs = filter_doc_list_through_topics(topics, docs)
    
    # Vectorise and TF-IDF transform the corpus 
    X, y, vectorizer = create_tfidf_training_data(ref_docs)
    
    # Create the training-test split of the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, \
                                                        test_size=0.2, \
                                                        random_state=42)
    
    # Create and train the Support Vector Machine
    svm = train_svm(X_train, y_train)
    # Save data from svm
    print "Saving data from training..."
    joblib.dump(svm, "saved_data/saved_trained_data.pkl")
    joblib.dump(vectorizer, "saved_data/saved_tfidfvectorizer_instance.pkl")
    joblib.dump(X, "saved_data/saved_tf_idf_weighted_document_term_matrix_from_tfidfvectorizer.pkl")

    # Make an array of predictions on the test set
    pred = svm.predict(X_test)

    # Output the hit-rate and the confusion matrix for each model
    print "\nHit-rate score is ", svm.score(X_test, y_test)

    """ By definition a confusion matrix C is such that C(i, j) is 
    equal to the number of observations known to be in group i, 
    but predicted to be in group j.
    """
    print "Confusion matrix"
    print(confusion_matrix(pred, y_test))
    
    print "\nTotal runtime %.2f seconds." % (time.time() - start_time)
    


if __name__ == "__main__":
    main()
    
