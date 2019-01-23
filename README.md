# NLTKBayes
We are given a set of emails and we need to tag the speaker (lecturer). Some emails give a hint by adding this in a speakers attribute in the email header whereas some do not. This network will help identify speakers within the main body of an email.

To extract this, I have built a Naïve Bayes classifier (speaker or not speaker; binary output). (Since the Stanford tagger can identify people); given a list of names, how do we know it is the speaker?
Answer; verbs used to describe the speaker are usually different to verbs used in non-speaker sentences. (Pr(speaker | [verbs]))

Building the classifier

![Visual Representation](https://github.com/minhalkhan/NLTKBayes/blob/master/Screenshot%202019-01-23%20at%2012.40.52.png)

Steps in building the classifier (see Bayes.py):
* Load the training emails
* Get the speakers name by looking at the speaker tags (done using a regex
```
<speaker>(.*?)<\/speaker>)
```
* Run StanfordNERTagger to get all names within the email
* The Stanford tagger will pick up the speaker’s name, therefore we will
remove it from the list of names thereby only giving names that is not the
speaker
* Get all of the sentences in which the speaker is in
* For each sentence get all the verbs (I done this using Chunking with the
expression 
```
Verb: {<VB.?>*}
```
* I will then lemmatize the verbs (e.g. presenting --> present) Once
lemmatized, we will push this word as a tuple with an outcome of True (since it appeared in a sentence where there is a speaker) onto our training set
* We do the same but on the list of names we found that are not the speaker (and put the verbs with the outcome being False)
* We do this for every file within the training folder (0-300) and store the verb array in a .txt file (called ‘Bayes.txt’)
* At this point we have built our training data for our classifier, we just have to feed this into nltk.NaiveBayesClassifier and store the classifier (called bayes_classifier.pickle’)
Once we do this, we can classify if a list of verbs in a sentence is a speaker or not:
