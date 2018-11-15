import numpy as np


# In[6]:


def sigmoid(x):
    return 1/float((1+np.exp(-x)))

def sigmoidd(x):
    sig=[]
    for row in x:
        r=[]
        for value in row:
            r.append(sigmoid(value))
        sig.append(r)
    return sig

def diff_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def diff_sigmoidd(x):
    sig=[]
    for row in x:
        r=[]
        for value in row:
            r.append(diff_sigmoid(value))
        sig.append(r)
    return sig


# In[7]:


def tanh(x):
    return np.tanh(x)

def tanhh(x):
    tan=[]
    for row in x:
        r=[]
        for value in row:
            r.append(tanh(value))
        tan.append(r)
    return tan

def diff_tanh(x):
    return 1-x**2

def diff_tanhh(x):
    diftan=[]
    for row in x:
        r=[]
        for value in row:
            r.append(diff_tanh(value))
        diftan.append(r)
    return diftan


# In[4]:


def relu(x):
    return np.log(1+np.exp(x))

def reluu(x):
    rel=[]
    for row in x:
        r=[]
        for value in row:
            r.append(relu(x))
        rel.append(r)
    return rel

def diff_relu(x):
    return 1/float(1+np.exp(-x))

def diff_reluu(x):
    difrel=[]
    for row in x:
        r=[]
        for value in row:
            r.append(diff_relu(x))
        rel.append(r)
    return rel



#Gaussian

def gauss(x):
    return np.exp(-1*x**2)

def gausss(x):
    gau=[]
    for row in x:
        r=[]
        for value in row:
            r.append(gauss(value))
        gau.append(r)
    return gau

def diff_gauss(x):
    return -2*x*np.exp(-1*x**2)

def diff_gausss(x):
    difgau=[]
    for row in x:
        r=[]
        for value in row:
            r.append(diff_gauss(value))
        difgau.append(r)
    return difgau