

def __phi__(phi_0, phi_1):
    if phi_0:
        return phi_0
    return phi_1

def set_field_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base

def set_index_wrapper(base, attr, value):
    setattr(base, attr, value)
    return base

def global_wrapper(x):
    return x
import pandas as pd
from MongoClient import read_mongo
import numpy as np
from textblob import TextBlob
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
_var0 = 'english'
cachedStopWords = stopwords.words(_var0)
algorithm = 'multinomialnb'

def splitIntoTokens(message):
    _var1 = TextBlob(message)
    _var2 = _var1.words
    return _var2

def splitIntoLemmas(message_0):
    message_1 = message_0.lower()
    _var3 = TextBlob(message_1)
    words = _var3.words
    _var4 = [word.lemma for word in words]
    return _var4

def hasUrl(message_2):
    _var5 = '(http://+)|(www+)'
    r = re.compile(_var5)
    match = r.search(message_2)
    _var6 = None
    _var7 = (match is _var6)
    if _var7:
        _var8 = 0
        return _var8
    _var9 = 1
    return _var9

def splitIntoWords(text):
    _var10 = 'html.parser'
    _var11 = BeautifulSoup(text, _var10)
    textNoHtml = _var11.get_text()
    _var12 = '[^a-zA-Z]'
    _var13 = ' '
    lettersOnly = re.sub(_var12, _var13, textNoHtml)
    _var14 = lettersOnly.lower()
    words_0 = _var14.split()
    _var15 = global_wrapper(cachedStopWords)
    stops = set(_var15)
    woStopWords = [word for word in words if (not (word in stops))]
    _var16 = ' '
    _var17 = _var16.join(woStopWords)
    baseForm = splitIntoLemmas(_var17)
    _var18 = ' '
    _var19 = _var18.join(baseForm)
    return _var19

def searchBestModelParameters(algorithm_0, trainingData):
    _var20 = 'multinomialnb'
    _var21 = (algorithm_0 == _var20)
    if _var21:
        alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
        fitPrior = [True, False]
        paramDistribution = dict(alpha=alpha, fit_prior=fitPrior)
        model = MultinomialNB()
    bestRun = []
    _var22 = 1
    _var23 = range(_var22)
    for _ in _var23:
        _var24 = 10
        _var25 = 'precision'
        _var26 = 5
        rand = RandomizedSearchCV(model, paramDistribution, cv=_var24, scoring=_var25, n_iter=_var26)
        _var27 = 'isSpam'
        _var28 = trainingData[_var27]
        rand_0 = rand.fit(trainingData, _var28)
        _var29 = rand_0.best_score_
        _var30 = 3
        _var31 = round(_var29, _var30)
        _var32 = rand_0.best_params_
        _var33 = {'score': _var31, 'params': _var32}
        bestRun.append(_var33)

    def _func0(x):
        _var34 = 'score'
        _var35 = x[_var34]
        return _var35
    _var36 = max(bestRun, key=_func0)
    print(_var36)

    def _func1(x_0):
        _var37 = 'score'
        _var38 = x_0[_var37]
        return _var38
    _var39 = max(bestRun, key=_func1)
    return _var39

def predictAndReport(algo, train, test, bestParams=None):
    _var40 = 'multinomialnb'
    _var41 = (algo == _var40)
    if _var41:
        _var42 = True
        _var43 = 0.8
        predictor = MultinomialNB(fit_prior=_var42, alpha=_var43)
    _var44 = 'isSpam'
    _var45 = train[_var44]
    predictor_0 = predictor.fit(train, _var45)
    predicted = predictor_0.predict(test)
    _var46 = ['predictedClass']
    dfWithClass = pd.DataFrame(predicted, columns=_var46)
    _var47 = [test, dfWithClass]
    _var48 = 1
    final = pd.concat(_var47, axis=_var48)
    _var49 = final.isSpam
    _var50 = final.predictedClass
    _var51 = pd.crosstab(_var49, _var50)
    print(_var51)
    _var52 = '0s: %d, 1s: %d'
    _var53 = final.isSpam
    _var54 = 0
    _var55 = (_var53 == _var54)
    _var56 = final.predictedClass
    _var57 = 0
    _var58 = (_var56 == _var57)
    _var59 = (_var55 & _var58)
    _var60 = np.sum(_var59)
    _var61 = final.isSpam
    _var62 = 1
    _var63 = (_var61 == _var62)
    _var64 = final.predictedClass
    _var65 = 1
    _var66 = (_var64 == _var65)
    _var67 = (_var63 & _var66)
    _var68 = np.sum(_var67)
    _var69 = (_var60, _var68)
    _var70 = (_var52 % _var69)
    print(_var70)
    _var71 = 'Accuracy: %.3f'
    _var72 = final.isSpam
    _var73 = final.predictedClass
    _var74 = (_var72 == _var73)
    _var75 = np.sum(_var74)
    _var76 = len(test)
    _var77 = float(_var76)
    _var78 = (_var75 / _var77)
    _var79 = float(_var78)
    _var80 = (_var71 % _var79)
    print(_var80)
    _var81 = 'Precision: %.3f'
    _var82 = final.isSpam
    _var83 = 1
    _var84 = (_var82 == _var83)
    _var85 = final.predictedClass
    _var86 = 1
    _var87 = (_var85 == _var86)
    _var88 = (_var84 & _var87)
    _var89 = np.sum(_var88)
    _var90 = final.isSpam
    _var91 = 1
    _var92 = (_var90 == _var91)
    _var93 = np.sum(_var92)
    _var94 = (_var89 / _var93)
    _var95 = float(_var94)
    _var96 = (_var81 % _var95)
    print(_var96)
_var97 = 'CB'
_var98 = 'journal'
_var99 = 'localhost'
rawJournals = read_mongo(db=_var97, collection=_var98, host=_var99)
_var100 = 'body'
_var101 = rawJournals[_var100]
_var102 = list(_var101)
_var103 = ['content']
journals = pd.DataFrame(_var102, columns=_var103)
_var104 = 'siteId'
_var105 = 'siteId'
_var106 = rawJournals[_var105]
journals_0 = set_index_wrapper(journals, _var104, _var106)
_var107 = 'text'
_var108 = 'title'
_var109 = rawJournals[_var108]
_var110 = _var109.astype(str)
_var111 = ' '
_var112 = (_var110 + _var111)
_var113 = 'content'
_var114 = journals_0[_var113]
_var115 = (_var112 + _var114)
journals_1 = set_index_wrapper(journals_0, _var107, _var115)
_var116 = ['content']
_var117 = True
_var118 = 1
journals_1.drop(_var116, inplace=_var117, axis=_var118)
_var119 = 'CB'
_var120 = 'site'
_var121 = 'localhost'
_var122 = False
rawSite = read_mongo(db=_var119, collection=_var120, host=_var121, no_id=_var122)
_var123 = '_id'
_var124 = rawSite[_var123]
_var125 = list(_var124)
_var126 = ['siteId']
siteIds = pd.DataFrame(_var125, columns=_var126)
_var127 = 'isSpam'
_var128 = 'isSpam'
_var129 = rawSite[_var128]
siteIds_0 = set_index_wrapper(siteIds, _var127, _var129)
_var130 = siteIds_0.isSpam
_var131 = 0
_var132 = True
_var130.fillna(_var131, inplace=_var132)
_var133 = {'isSpam': 'isSiteSpam'}
_var134 = True
siteIds_0.rename(columns=_var133, inplace=_var134)
_var135 = '/Users/dmurali/Documents/spamlist_round25_from_20150809_to_20151015.csv'
_var136 = ['siteId', 'isSpam']
octSiteProfileSpam = pd.read_csv(_var135, usecols=_var136)
_var137 = {'isSpam': 'isOctSpam'}
_var138 = True
octSiteProfileSpam.rename(columns=_var137, inplace=_var138)
_var139 = 'left'
_var140 = ['siteId']
_var141 = False
_var142 = journals_1.merge(siteIds_0, how=_var139, on=_var140, sort=_var141)
_var143 = 'left'
_var144 = ['siteId']
_var145 = False
journalsFinal = _var142.merge(octSiteProfileSpam, how=_var143, on=_var144, sort=_var145)
_var146 = 'isSpam'
_var147 = 'isOctSpam'
_var148 = journalsFinal[_var147]
_var149 = 'isSiteSpam'
_var150 = journalsFinal[_var149]
_var151 = _var148.isin(_var150)
_var152 = 1
_var153 = 'isSiteSpam'
_var154 = journalsFinal[_var153]
_var155 = np.where(_var151, _var152, _var154)
journalsFinal_0 = set_index_wrapper(journalsFinal, _var146, _var155)
_var156 = ['isOctSpam', 'isSiteSpam']
_var157 = True
_var158 = 1
journalsFinal_0.drop(_var156, inplace=_var157, axis=_var158)
_var159 = 'text'
_var160 = journalsFinal_0[_var159]
_var161 = ' '
_var162 = True
_var160.fillna(_var161, inplace=_var162)
_var163 = 'text'
_var164 = 'text'
_var165 = journalsFinal_0[_var164]
_var166 = _var165.apply(splitIntoWords)
journalsFinal_1 = set_index_wrapper(journalsFinal_0, _var163, _var166)
_var167 = 'length'
_var168 = 'text'
_var169 = journalsFinal_1[_var168]

def _func2(text_0):
    _var170 = len(text_0)
    return _var170
_var171 = _var169.map(_func2)
journalsFinal_2 = set_index_wrapper(journalsFinal_1, _var167, _var171)
_var172 = CountVectorizer()
_var173 = 'text'
_var174 = journalsFinal_2[_var173]
wordsVectorizer = _var172.fit(_var174)
_var172_0 = wordsVectorizer
_var175 = 'text'
_var176 = journalsFinal_2[_var175]
wordsVector = wordsVectorizer.transform(_var176)
_var177 = TfidfTransformer()
inverseFreqTransformer = _var177.fit(wordsVector)
_var177_0 = inverseFreqTransformer
invFreqOfWords = inverseFreqTransformer.transform(wordsVector)
_var178 = invFreqOfWords.toarray()
weightedFreqOfWords = pd.DataFrame(_var178)
_var179 = 'isSpam'
_var180 = 'isSpam'
_var181 = journalsFinal_2[_var180]
weightedFreqOfWords_0 = set_index_wrapper(weightedFreqOfWords, _var179, _var181)
_var182 = 'isSpam'
_var183 = 'isSpam'
_var184 = weightedFreqOfWords_0[_var183]
_var185 = _var184.astype(int)
weightedFreqOfWords_1 = set_index_wrapper(weightedFreqOfWords_0, _var182, _var185)
_var190 = 'isSpam'
_var191 = weightedFreqOfWords_1[_var190]
_var192 = 0.5
(_var186, _var187, _var188, _var189) = train_test_split(weightedFreqOfWords_1, _var191, test_size=_var192)
train_0 = _var186
test_0 = _var187
spamLabelTrain = _var188
spamLabelTest = _var189
predictAndReport(algo=algorithm, train=train_0, test=test_0)
