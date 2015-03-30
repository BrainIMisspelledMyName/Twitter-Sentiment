# -*- coding: utf-8 -*-

# Final Project
# CS175, Winter 2015
#
# Sentiment Analysis on Tweets 
#
# Authors:
#  Bing Hui Feng, 78912993, bhfeng@uci.edu 
#  Jessica Pham, 80859856, jessictp@uci.edu
#  Brian Chou, 72312823, bchou1@uci.edu

"""
    PROGRAM DESCRIPTION
        this script runs the three classifiers on data preprocessed by datapreprocess.py
        and prints the scores of each test to a file 
"""

#preprocessing
import pandas as pd
from nltk.corpus import stopwords
import datapreprocess

#feature extraction
from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfTransformer
from collections import namedtuple

# learning algorithms and validation methods
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from statistics import mean;

#defining namedtuples
PredictionsInfo = namedtuple("PredictionsInfo", "data_desc NB_tr_cv LR_tr_cv SVM_tr_cv NB_te_cv LR_te_cv \
                                         SVM_te_cv NB_tr LR_tr SVM_tr  NB_te LR_te SVM_te")
PredictionsInfoLess = namedtuple("PredictionsInfoLess", "data_desc NB_tr_cv LR_tr_cv SVM_tr_cv NB_te_cv LR_te_cv \
                                         SVM_te_cv")
dataSet = namedtuple("dataSet", "Xtr Ytr Xte Yte")

                                       
"""
    load_data
        takes the csv files loacted at http://help.sentiment140.com/for-students/
        and creates term frequency inverse documnet frequency object with the set number of grams
"""

def load_data(train_csv, test_csv, numGramStart:int, numGramEnd:int):
    print('creating grams ranging ', numGramStart , "to", numGramEnd)
    print("processing data from csv to data matrix....")
    input_twitter_test_data = pd.read_csv(test_csv, encoding='ISO-8859-1',error_bad_lines=False);
    input_twitter_training_data = pd.read_csv(train_csv, encoding='ISO-8859-1',error_bad_lines=False);
    
    #prepare training data
    train_twitter_data = input_twitter_training_data.as_matrix();
    train_tweets = train_twitter_data[:,1]; # numpy array
    Y_train = train_twitter_data[:,0].astype('str');
    
    cv = CountVectorizer()
    cv.ngram_range = (numGramStart,numGramEnd) 
    cv.stop_words = stopwords.words('english')
    tfidf_transformer = TfidfTransformer()
    #tf_transformer = TfidfTransformer(use_idf=False)
    
    print('constructing featuresets....');
    X_train_counts = cv.fit_transform(train_tweets)
    #X_train_tf = tf_transformer.fit_transform(X_train_counts);
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts) 
    print('DONE....');
    
    #prepare test data
    
    print('preparing test data....')
    test_twitter_data = input_twitter_test_data.as_matrix();    
    test_tweets = test_twitter_data[:,1];
    Y_test = test_twitter_data[:,0].astype('str')
    
    X_test_counts = cv.transform(test_tweets);
    #X_test_tf = tf_transformer.transform(X_test_counts);
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    print("DONE....")
    print()
    
    return dataSet(X_train_tfidf, Y_train, X_test_tfidf, Y_test)
                                       
"""
    cross_validate
        divides the test data of both X and Y .75/.25 training/test split
        
"""
def cross_validate(X_train_data, Y_train_data, classifier, k , seed):    
    # local varibales
    TrainAccuracies = [];
    TestAccuracies = [];
    size = X_train_data.shape
    # shuffles the content using stratified split. "Even split of the two classes"
    sss = cross_validation.StratifiedShuffleSplit(X_train_data, k, test_size=.25, random_state=seed)
    
    for train_indices, test_indices in sss:
        #partition the data    
        X_train_parition, X_test_parition = X_train_data[train_indices], X_train_data[test_indices]
        Y_train_parition, Y_test_parition = Y_train_data[train_indices], Y_train_data[test_indices]
            
        c = classifier.fit(X_train_parition, Y_train_parition)
        
        trscore = c.score(X_train_parition,Y_train_parition)     
        tescore = c.score(X_test_parition,Y_test_parition)
        TrainAccuracies.append(trscore)
        TestAccuracies.append(tescore)
        
    return TrainAccuracies, TestAccuracies
    
"""
    runTests
        runs cross validate on the 3 different classifiers and returns the training and testing
        accuracies for each
"""
def runTests(X_train, Y_train, X_test, Y_test, classifiers, data_desc):
        
    print('computing cross validation....')
    K = 5
    seed = 1;
    # cross-validate returns 6 lists of values that correspond to each classifier's training and test accuracies based on the number of folds (which is k)
    NBTrainAccuracies , NBTestAccuracies = cross_validate(X_train, Y_train, classifiers[0], K, seed)
    LRTrainAccuracies , LRTestAccuracies = cross_validate(X_train, Y_train, classifiers[1], K, seed)
    SVMTrainAccuracies , SVMTestAccuracies = cross_validate(X_train, Y_train, classifiers[2], K, seed)
    print('DONE.... \n')
    # print values the mean of the cross validated accuracies
    print("cross validation results \n")
    print("NB CV Train mean: ", mean(NBTrainAccuracies))
    print("LR CV Train mean: ", mean(LRTrainAccuracies))
    print("SVM CV Train mean: ", mean(SVMTrainAccuracies))
    print();
    print("NB CV Test mean: ", mean(NBTestAccuracies))
    print("LR CV Test mean: ", mean(LRTestAccuracies))
    print("SVM CV Test mean: ", mean(SVMTestAccuracies))
    
    
    #everything should have been printed save for the lists.    
    return PredictionsInfoLess(data_desc, NBTrainAccuracies,LRTrainAccuracies,SVMTrainAccuracies, 
                                          NBTestAccuracies, LRTestAccuracies, SVMTestAccuracies )
    # last method we are using is support vector machines

"""
    create_classifiers
        creates three object classifers, logistic regression, Multinomial Naive Bayes
        and linear SVM. Here we changed thier parameters
"""

def create_classifiers():

    # these are the various parameters we used for different runs
    # Default run -> comment out the nb.alpha, LR.C, svm.C
    # diffParam(higherRegulization) = nb.alpha = 1.2, LR.8, SVM.8
    # lowRegularization = nb.alpha= .8, lr.c = 1.2, svm = 1.2
    # also change  the line below to make the text output a descriptiob that fits the learner changes

    classifiers = []
    nb = MultinomialNB();
    #nb.alpha = 1.2
    lr = LogisticRegression()
    #lr.C = .8
    svm = LinearSVC()
    #svm.C = .8
    classifiers.append(nb)
    classifiers.append(lr)
    classifiers.append(svm)

    return classifiers
    
"""
    create_dataset
        creates the datasets that is used by cross validate
"""
def create_dataset(nopreprocessTrainPath, nopreprocessTestPath, preprocessedTrainPath, preprocessedTestPath):
    #create datasets depending on the number of grams
    dataSets = []
    # unigrams
    dataSets.append(load_data(nopreprocessTrainPath, nopreprocessTestPath,1,1))
    dataSets.append(load_data(preprocessedTrainPath, preprocessedTestPath,1,1))
    
    # unigrams and bigrams
    dataSets.append(load_data(nopreprocessTrainPath, nopreprocessTestPath,1,2))
    dataSets.append(load_data(preprocessedTrainPath, preprocessedTestPath,1,2))
    
    # bigrams
    dataSets.append(load_data(nopreprocessTrainPath, nopreprocessTestPath,2,2))
    dataSets.append(load_data(preprocessedTrainPath, preprocessedTestPath,2,2))
    return dataSets
 
"""
    get_results
        runs each of the tests on all data within the dataset made from create_dataset
        returns the result to the main to be printed out to a file
"""
def get_results(dataSets, classifiers ,parameters ):
        
    # testing each unigrams, bigrams and both for all tests and classifiers
    # list of PredictionInfos
    results = [] 
    
    #unigrams testing    
    print("NO PREPROCESSING unigrams " + parameters)
    results.append(runTests(dataSets[0].Xtr,dataSets[0].Ytr,dataSets[0].Xte, dataSets[0].Yte,classifiers, 
                            ("NO PREPROCESSING unigrams " + parameters)))    

    print("\n\nPRE PROCESSED unigrams " + parameters)
    results.append(runTests(dataSets[1].Xtr,dataSets[1].Ytr,dataSets[1].Xte, dataSets[1].Yte,classifiers,
                            ("PRE PROCESSED unigrams " + parameters)))  

    #unigrams and bigrams testing
    print("\n\n\nNO PREPROCESSING unigrams and bigrams " + parameters)   
    results.append(runTests(dataSets[2].Xtr,dataSets[2].Ytr,dataSets[2].Xte, dataSets[2].Yte,classifiers,
                            ("NO PREPROCESSING unigrams and bigrams " + parameters)))  

    print("\n\nPRE PROCESSED unigrams and bigrams " + parameters)  
    results.append(runTests(dataSets[3].Xtr,dataSets[3].Ytr,dataSets[3].Xte, dataSets[3].Yte,classifiers,
                            ("PRE PROCESSED unigrams and bigrams " + parameters)))  

    #bigrams testing
    print("\n\n\nNO PREPROCESSING bigrams " + parameters)     
    results.append(runTests(dataSets[4].Xtr,dataSets[4].Ytr,dataSets[4].Xte, dataSets[4].Yte,classifiers,
                            ("NO PREPROCESSING bigrams " + parameters)))  

    print("\n\nPRE PREPROCESSED bigrams " + parameters) 
    results.append(runTests(dataSets[5].Xtr,dataSets[5].Ytr,dataSets[5].Xte, dataSets[5].Yte,classifiers,
                            ("PRE PREPROCESSED bigrams " + parameters))) 
    return results
    
"""
    do_classification
        the main part of the program that runs all the auxillary function to create and the classifers
        and copmute thier accuracies
"""
def do_classification():
    #file locations   
    testfile, trainfile  = 'data/testdata.csv' , 'data/training.csv'
    nopreprocessTestPath, nopreprocessTrainPath  = 'data/neutralsRemovedOnlyTest.csv' , 'data/neutralRemovedOnlyTrain.csv'
    preprocessedTestPath, preprocessedTrainPath = 'data/neutralsRemovedAndPreOnTest.csv' , 'data/neutralsRemovedAndPreOnTrain.csv'
    
    #preprocess data if not yet preprocessed
    datapreprocess.preprocess_data(testfile, nopreprocessTestPath, False, False, False, True)
    datapreprocess.preprocess_data(trainfile, nopreprocessTrainPath, False, False, False, True)
    datapreprocess.preprocess_data(testfile, preprocessedTestPath , True, True, True, True)
    datapreprocess.preprocess_data(trainfile, preprocessedTrainPath, True, True, True, True) 

    dataSets = create_dataset(nopreprocessTrainPath, nopreprocessTestPath, preprocessedTrainPath, preprocessedTestPath)

    parameters = "nb.alpha = 1.2, LR.8, SVM.8"
    classifiers = create_classifiers()
    
    results = get_results( dataSets, classifiers , parameters )
    
    return results
    
"""
   write_data_tofile
        writes all the data from results form do_classifcation and prints it to a text file 
"""    
def write_data_tofile(results):
    f = open('results/ResultsDefault.txt','w')
    for result in results:
        f.write(result.data_desc + '\n \n')
        
        f.write("cross validation NB folds \n")
        f.write(' TRAIN            TEST\n')
        for i in range(len(result.NB_tr_cv)):
            f.write (str(result.NB_tr_cv[i])  + '  ' + str(result.NB_te_cv[i]) + '\n')  
        f.write('\n')
        
        f.write("cross validation LR folds \n")
        f.write(' TRAIN            TEST\n')
        for i in range(len(result.LR_tr_cv)):
            f.write (str(result.LR_tr_cv[i])  + '  ' + str(result.LR_te_cv[i]) + '\n')
        f.write('\n')
        
        f.write("cross validation SVM folds \n")
        f.write(' TRAIN            TEST \n')
        for i in range(len(result.SVM_tr_cv)):
            f.write (str(result.SVM_tr_cv[i])  + '  ' + str(result.SVM_te_cv[i]) + '\n')
        f.write('\n')
        
        f.write("cross validation results \n")
        f.write("NB CV means: \n" + " TRAIN: "+ str(mean(result.NB_tr_cv)) + "   TEST: " + str(mean(result.NB_te_cv)) + " \n")
        f.write("LR CV means: \n" + " TRAIN: "+ str(mean(result.LR_tr_cv)) + "   TEST: " + str(mean(result.LR_te_cv)) + " \n")
        f.write("SVM CV means: \n" + " TRAIN: "+ str(mean(result.SVM_tr_cv)) + "   TEST: " + str(mean(result.SVM_te_cv)) + " \n")
        f.write('\n \n')
        f.flush()
        
    f.close()

    
if __name__ == "__main__":
#    do_classification()
    results = do_classification()
    write_data_tofile(results)
    print("PROGRAM COMPLETED")