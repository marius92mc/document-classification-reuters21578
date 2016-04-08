import time
import json

from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer


def test_data_from_json_file():
    start_time = time.time()
    # Load training data from files
    print "Load training data..."
    svm = joblib.load("saved_data/saved_trained_data.pkl")
    count_vect = joblib.load("saved_data/saved_tfidfvectorizer_instance.pkl")
    X_train_counts = joblib.load("saved_data/saved_tf_idf_weighted_document_term_matrix_from_tfidfvectorizer.pkl")
    
    # Open files with test data, in json format
    print "Load testing data..."
    test_file = open("data/reuters_test_json/reuters_test1.json", "r")
    data = json.load(test_file)
    
    # Load specific fields from json data
    print "Parsing testing data..." 
    corpus_test = [entry["body"] for entry in data]
    corpus_ids = [int(entry["id"]) for entry in data]
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    X_new_counts = count_vect.transform(corpus_test)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    predicted = svm.predict(X_new_tfidf)
    #print y
    #print predicted
    ''' predicted[i] = predicted value for corpus_test[i], 
                            where corpus_test has data from test json
    for predicted_value in predicted:
        print predicted_value.encode("utf-8")
    '''
    print "\n id  Predicted topic"
    print "---- ---------------"
    for corpus_id, predicted_value in zip(corpus_ids, predicted):
        print "%4d %s" % (corpus_id, predicted_value.encode("utf-8"))
    
    y_test = []
    for entry in data:
        if "topics" in entry:
            for topic in entry["topics"]:
                if len(topic) > 0:
                    y_test.append(topic.encode("utf-8")) 
                    break # put only the first topic for the specified article 

    if len(y_test) == len(predicted):
        print "\nHit-rate score is ", svm.score(X_new_counts, y_test)
        print "Confusion matrix"
        print confusion_matrix(predicted, y_test)
    print "\nTotal runtime %.2f seconds." % (time.time() - start_time)


def get_all_topics_from_trained_json_data():
    all_topics = []
    
    for file_name in ["data/reuters_json/reuters-%03d.json" % r for r in range(0, 22)]:
        json_data = open(file_name, "r")
        data = json.load(json_data)
        for json_entry in data:
            if "topics" in json_entry:
                for topic in json_entry["topics"]:
                    all_topics.append(topic)
    
    hash_table = {}
    for topic in all_topics:
        if topic not in hash_table:
            hash_table[topic] = 1
        else:
            hash_table[topic] += 1
            
    for key in hash_table:
        print key, hash_table[key]
        


    
test_data_from_json_file()

#get_all_topics_from_trained_json_data()

