#!/usr/bin/python
import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import precision, recall, f_measure
############################################################################
#                                                                          #
#                   Author: William Tsoi                                   #
#                   Course: CKME 136 Capstone Project - Ryerson University #
#                   Date  : 27 Mar 2016                                    #
#                   Acknowledgement : Special thanks to Andy Bromberg      #
#                   of andybromberg.com  and Laurent Luce of               #
#                   laurentluce.com for their knowledge contribution       #
#                   and code sharing on their respective websites.         #
#                                                                          #
############################################################################
#     
#
# Define prelabelled tweets with sentiments for training and testing
# Positive sentinment file polarity-pos.txt
RT_POLARITY_POS_FILE = "d:/test/polarity-pos.txt"
# Negative sentinment file polarity-neg.txt
RT_POLARITY_NEG_FILE = "d:/test/polarity-neg.txt"

#creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])
	
def evaluate_features(feature_select):

 
    posFeatures = []
    negFeatures = []
	#
    #breaks up the sentences into lists of individual words (as selected by the input mechanism) and appends 'pos' or 'neg' after each list
	#
    with open(RT_POLARITY_POS_FILE, 'r') as posSentences:
        for i in posSentences:
			# strip out embedded URLs in the tweets
            j = re.sub(r'(https?:\/\/)+([\da-z\.-]+)\/([a-zA-Z\.0-9]{1,})', "", i)
			# Return a list of strings
            posWords = re.findall(r"[\w']+|[.,!?;]", j.rstrip())
			# Convert strings to lower case and only accept strings with 3 or more characters
            posWords_filtered = [e.lower() for e in posWords if len(e) >= 3] 
			# Mark the sentiment as positive
            posWords_filtered = [feature_select(posWords_filtered), 'pos']
            posFeatures.append(posWords_filtered)
    with open(RT_POLARITY_NEG_FILE, 'r') as negSentences:
        for i in negSentences:
			# strip out embedded URLs in the tweets
            j = re.sub(r'(https?:\/\/)+([\da-z\.-]+)\/([a-zA-Z\.0-9]{1,})', "", i)
			# Return a list of strings
            negWords = re.findall(r"[\w']+|[.,!?;]", j.rstrip())
			# Convert strings to lower case and only accept strings with 3 or more characters
            negWords_filtered = [e.lower() for e in negWords if len(e) >= 3]
			# Mark the sentiment as negative
            negWords_filtered = [feature_select(negWords_filtered), 'neg']
            negFeatures.append(negWords_filtered)

		
    posSentences.close
    negSentences.close



    #selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    #trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)	

    #initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
	
    testSets = collections.defaultdict(set)	

    #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)	

    #prints metrics to show how well the feature selection did
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', recall(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(10)
    print "********************************************************\n"
	#
	# Evaluate sentinment for Tesla first day of data collection Feb 29
	#
    TWITTER_DATA_FILE = "d:/test/tsla29Feb_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
    #
	# Evaluate sentinment for Tesla seond day of data collection Mar 01
	#
    TWITTER_DATA_FILE = "d:/test/tsla01Mar_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
	#
	# Evaluate sentinment for Tesla third day of data collection Mar 02
	#
    TWITTER_DATA_FILE = "d:/test/tsla02Mar_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
	#
	# Evaluate sentinment for Tesla fourth day of data collection Mar 03
	#
    TWITTER_DATA_FILE = "d:/test/tsla03Mar_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
	#
	# Evaluate sentinment for Tesla fifth day of data collection Mar 04
	#
    TWITTER_DATA_FILE = "d:/test/tsla04Mar_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
	#
	# Evaluate sentinment for Tesla fifth day of data collection Mar 18
	#
    TWITTER_DATA_FILE = "d:/test/tsla18Mar_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
	#
	# Evaluate sentinment for Tesla fifth day of data collection Mar 21
	#
    TWITTER_DATA_FILE = "d:/test/tsla21Mar_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
	#
	# Evaluate sentinment for Tesla fifth day of data collection Mar 22
	#
    TWITTER_DATA_FILE = "d:/test/tsla22Mar_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
	#
	# Evaluate sentinment for Tesla fifth day of data collection Mar 23
	#
    TWITTER_DATA_FILE = "d:/test/tsla23Mar_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
	#
	# Evaluate sentinment for Tesla fifth day of data collection Mar 24
	#
    TWITTER_DATA_FILE = "d:/test/tsla24Mar_formatted.csv"
    evaluate_twitter_data(TWITTER_DATA_FILE,make_full_dict,classifier)
	#
def evaluate_twitter_data(TWITTER_DATA_FILE,feature_select,classifier):	
	TwFeatures = []
	print "\nProcessing Data File %s\n" %TWITTER_DATA_FILE
	
	with open(TWITTER_DATA_FILE, 'r') as Twitter_Sentences:
         for i in Twitter_Sentences:
			# strip out embedded URLs
            j = re.sub(r'(https?:\/\/)+([\da-z\.-]+)\/([a-zA-Z\.0-9]{1,})', "", i)
			# Return a list of strings
            TwWords = re.findall(r"[\w']+|[.,!?;]", j.rstrip())
			# Convert strings to lower case and only accept strings with 3 or more characters
            TwWords_filtered = [e.lower() for e in TwWords if len(e) >= 3] 
            TwWords_filtered = [feature_select(TwWords_filtered),'test']
			# Append to the Feature list
            TwFeatures.append(TwWords_filtered)
        Twitter_Sentences.close
	
	referenceSets = collections.defaultdict(set)
	testSets = collections.defaultdict(set)	
	posCount = 0
	negCount = 0
	ambiguousCount = 0
	#
	# Process the daily data
	#
	for i, (features,label) in enumerate(TwFeatures):
		referenceSets[label].add(i)
		# Process the tweet to confirm whether it is positive or negative
		predicted = classifier.classify(features)
		testSets[predicted].add(i)	
		if predicted == 'pos':
			posCount +=  1
		elif predicted == 'neg':
			negCount +=  1
		else:
			print "Ambiguous Status>>> %s" %predicted
			ambiguousCount +=1


	print "Total Positive Count %d" %posCount
	print "Total Negative Count %d" %negCount
	print "Total Ambiguous Count %d" %ambiguousCount
	
	if posCount >= negCount :
		print "The Sentinment for %s is POSITIVE\n" %TWITTER_DATA_FILE
	else:
		print "The Sentinment for %s is NEGATIVE\n" %TWITTER_DATA_FILE
	print "********************************************************\n"

	
#creates a feature selection mechanism that uses all words
def make_full_dict(words):
	return dict([(word, True) for word in words])
	
print 'using all words as features'
evaluate_features(make_full_dict)

