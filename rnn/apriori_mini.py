
# coding: utf-8

# In[1]:


from numpy import *


# In[2]:


paragraph='Lambda functions can have any number of arguments but only one expression The expression is evaluated and returned. Lambda functions can be used wherever function objects are required'

paragraph='In conclusion I will add that after the natural qualifications for a good detective have developed themselves it takes more hard work and study to reach the pinnacle of fame than other professions require and the remuneration is a great deal less taking into consideration the hazardousness of the business'


# In[18]:


file=open('lzw_compression.txt',encoding="utf8")
text=[]
for line in file.readlines():
    sentence=line.rstrip()
    if(len(sentence)!=0):
        text.append(line.rstrip())


# In[19]:


para=[]
for line in text:
    para.append(line.split(' '))
print(para)


# In[21]:


print(paragraph.split(' '))


# In[22]:


text=[]
for words in paragraph.split(' '):
    text.append(list(words))


# In[11]:


text


# In[ ]:


def loadDataSet(paragraph, mode=0):
    if mode==1:
        text=[]
        for words in paragraph.split(' '):
            text.append(list(words))
        return text
    elif mode==0:
        words=paragraph.split(' ')
        words_list=[]
        for word in words:
            words_list.append(word)
        return words_list


# In[ ]:


dataSet_char = loadDataSet(paragraph,1)
dataSet_words= loadDataSet(paragraph,0)


# In[ ]:


dataSet_words


# In[ ]:


def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return list(map(frozenset, C1))#use frozen set so we
                            #can use it as a key in a dict 


# In[ ]:


createC1(dataSet_words)


# In[ ]:


def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData


# In[ ]:


len(dataSet_words)


# In[ ]:


i=0
leng=7
j=leng
real_list=[]
while(i<len(dataSet_words)):
    print(i,j)
    real_list.append(dataSet_words[i:j])
    i=j+1
    j+=leng
    if j>len(dataSet_words):
        j=len(dataSet_words)+1


# In[ ]:


k=-1
print(real_list[k])
print(len(real_list[k]))


# In[ ]:


test=real_list[0]


# In[ ]:


def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): 
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList


# In[ ]:


def apriori(dataSet, minSupport = 0):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)#scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# In[ ]:


def generateRules(L, supportData, minConf=0.7):  #supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):#only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         


# In[ ]:


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] #create new list to return
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] #calc confidence
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


# In[ ]:


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): #try further merging
        Hmp1 = aprioriGen(H, m+1)#create Hm+1 new candidates
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    #need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


# In[ ]:


L,suppData= apriori(dataSet_char,minSupport=0.1)
rules_char= generateRules(L,suppData, minConf=0.5)


# In[ ]:


suppData


# In[ ]:


def relevant_rules(rules):
    relevant_rules=[]
    right=[]
    for rule in rules:
        if len(rule[0])<=len(rule[1]) and (rule[1] not in right):
            relevant_rules.append(rule)
            right.append(rule[0])
    return relevant_rules
#print(len(relevant_rules))


# In[ ]:


relevant_rules(rules_char)


# In[ ]:


real_list


# In[ ]:


createC1(real_list)


# In[ ]:


L_words,suppData_words= apriori(dataSet_words,minSupport=0)
rules_words= generateRules(L_words,suppData_words, minConf=0)


# In[ ]:


relevant_rules(rules_words)


# In[ ]:


len(rules_words)

def dictionary(uncompressed):
	# Build the dictionary.
    dict_size = 256
    #dictionary = dict((chr(i), i) for i in range(dict_size))
    # in Python 3:
    dictionary = {chr(i): i for i in range(dict_size)}
 
    w = ""
    result = []
    for c in uncompressed:
        wc = w + c
        if wc in dictionary:
            w = wc
        else:
            result.append(dictionary[w])
            # Add wc to the dictionary.
            dictionary[wc] = dict_size
            dict_size += 1
            w = c
 
    # Output the code for w.
    if w:
        result.append(dictionary[w])
    return result