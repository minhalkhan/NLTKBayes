import nltk
import re
from nltk import NaiveBayesClassifier as nbc
from nltk.stem import WordNetLemmatizer 
from nltk.tag.stanford import StanfordNERTagger
import shelve
from nltk.tokenize import word_tokenize
from itertools import chain
import pickle

"""
'Presenting,Presentation' -> {Lemmatize} -> Present <- Run this on the Bayes model
x = Pr(present|speaker) = highly probable it will be the speaker
y = Pr(present|not speaker) = low probability
Argmax(x,y) -> The passed name being the speaker

a = Pr(host|speaker) = low probability of being the speaker
b = Pr(host|not speaker)) = highly probable it will NOT be the speaker
Argmax(a,b) -> The passed name NOT being the speaker

Pr(Speaker) = 1 / Number of Names generated (WE ONLY RUN BAYES IF WE HAVE MORE THAN ONE NAME)
"""
class NaiveBayes:

    #Used to extract names from the email 
    st = StanfordNERTagger('stanford-ner/english.all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')

    #Used to get the lem of a word 
    lemmatizer = WordNetLemmatizer() 

    #address where the training data is located
    trainingAddress = 'corpora/assignment1/training/'

    #Verbs that may be used to represent the speaker
    #(Verb Word: True)
    verbs = []   #create dictionary of verbs used to describe a speaker 

    #Start number of training text files
    startNo = 0

    #End number of training text files
    endNo = 301

    #Expression to extract the speakers name
    regEx = '<speaker>(.*?)<\/speaker>'

    #split paragraphs, then split it into sentences
    paraSplitter = '(<paragraph>(\n.*?|.*?)*<\/paragraph>)'

    #Tokenize sentences
    splitSentence = '(<sentence>(\n.*?|.*?)*<\/sentence>)'

    #Array of speaker names (e.g. Two Speakers in a lecture or more than one way of writing a speakers name)
    #We dont really care about the name but more of the wordings that point to that speaker (Verbs)
    speakers = None  #Iterator object 
    
    def __init__(self):
        f = open('bayes_classifier.pickle', 'rb')
        self.classifier = pickle.load(f) 
        f.close()
        self.train()
        
    def train(self):
        #c = open('bayes_classifier.pickle', 'wb')

        f = open('Bayes.txt','r')
        self.training_data = [(tup.strip().split(':')[0],tup.strip().split(':')[1]) for tup in f]
        f.close()
        
        self.vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in self.training_data]))
        self.feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in self.vocabulary},tag) for sentence, tag in self.training_data]
        #self.classifier = nbc.train(self.feature_set)

        #pickle.dump(self.classifier, c)
        #c.close()


    def isSpeaker(self,verbs):
        test_sentence = verbs
        featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in self.vocabulary}
        return(self.classifier.classify(featurized_test_sentence))

    def build(self):
        
        for i in range(300,0,-1):#for i in range(self.startNo, self.endNo):
            print(i)
            self.email = self.getEmail(i)
            self.getSpeakers()

            if self.speakerNames != set():
                #Speaker found
                self.splitEmail()
                #self.getAllNames()
                
                #Try getting sentence if it is in a sentence
                #check if speaker tag is in array   
                self.getSpeakerVerbs()

                if len(self.not_speaker_names) > 0:
                    #names exixts get verbs on those words
                    for non_speaker in self.not_speaker_names:
                        self.getNonSpeakerVerbs(non_speaker)

            #move to next file if no speaker tag was found

        #Store Verbs
        print(self.verbs)
        self.storeVerbs()

    def storeVerbs(self):
        f = open('Bayes.txt','w')
        f.write('\n'.join(self.verbs)) #longest lenght at the top to help tagging 
        f.close()

    def getNonSpeakerVerbs(self, name):
        for i in self.sentArray:
            if name in i:
                tokenized = nltk.word_tokenize(i)
                tagged = nltk.pos_tag(tokenized)
                self.verbPicker([tagged],False)

    def getSpeakerVerbs(self):
        for i in self.sentArray:
            if '<speaker>' in i:
                sent_without_tag = re.sub('<speaker>','',i)
                sent_without_tag = re.sub('</speaker>','',sent_without_tag)
                tokenized = nltk.word_tokenize(sent_without_tag)
                tagged = nltk.pos_tag(tokenized)
                self.verbPicker([tagged], True)
                
    def verbPicker(self,parseTree,booli):
        c = self.chunk(r"""Verb: {<VB.?>*}""",parseTree) #gets all verbs
        
        for i in c:
            for subtree in i.subtrees():
                if subtree.label() == "Verb": 
                    #Add into dictionary
                    word = subtree.leaves()[0][0].lower()
                    lemmedWord = self.lemmatizer.lemmatize(word,'v')
                    self.verbs.append(lemmedWord+':'+str(booli))
        

    def chunk(self,regex,on):
        parser = nltk.RegexpParser(regex)
        return [parser.parse(crnt) for crnt in on]

    def splitEmail(self):
        """
        print("splitting para")
        paraArray = re.split(self.paraSplitter,self.email)
        print(paraArray)
        #[paragraphs[sentences]]
        """
        sA = []
        sentArray = [re.split(self.splitSentence,self.email)]

        for p in range(0,len(sentArray)):
            for sent in range(0,len(sentArray[p])):
                crnt = sentArray[p][sent]
                if not '<paragraph>' in crnt:
                    #remove sentence tag
                    if '<sentence>' in crnt:
                        crnt = re.sub('<sentence>','',crnt)
                        crnt = re.sub('</sentence>','',crnt)

                    sA.append(crnt)
        
        self.sentArray = sA
        
    def getSpeakers(self):
        self.speakers = re.finditer(self.regEx, self.email)
        #{'Professor Cusumano', 'Michael A. Cusumano'}
        #i.span()[0]+9 gets start index of speaker since len(<speaker>) = 9
        self.speakerNames = set([self.email[i.span()[0]+9:i.span()[1]-10] for i in self.speakers])

    #Gets all names in Email taht is not the speaker
    def getAllNames(self):
        names = []
        for i in nltk.sent_tokenize(self.email):
            tokens = nltk.tokenize.word_tokenize(i)
            tags = self.st.tag(tokens)
            start = False
            name = ""
            for tag in tags:
                if tag[1] == 'PERSON':
                    if not start:
                        start = True
                    name = name + " " + tag[0]  
                else:
                    start = False
                    if name != "":
                        names.append(name)
                        name = ""
        
        #Remove Speker from Names
        not_speaker_names = []
        #['Michael|A.|Cusumano', 'Professor|Cusumano'] <- Used to capture different ways of getting the speaker using reg ex
        regSpeakerName = [x.replace(' ','|') for x in self.speakerNames]

        for i in names:
            speaker = False
            for exp in regSpeakerName:
                x = re.search(exp,i)
                if x != None:
                    #Found
                    speaker = True
                    break
            if not speaker:
                not_speaker_names.append(i)
        
        self.not_speaker_names = set(not_speaker_names) 

    def getEmail(self,number):
        return nltk.data.load(self.trainingAddress+str(number)+'.txt')

#Run to build files
#NaiveBayes().build()
