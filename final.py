import ANN_skeleton_tanh as ann_tanh
import ANN_skeleton_sigmoid as ann_sig
import numpy as np
import pickle
s=int(raw_input("Enter 1 for sigmoid or 2 for tanh\n"))
if s == 1:
    nn=ann_sig.NeuralNetwork(16*8*3,8,10)
    inputs=[]
    targets=[]
    a1=pickle.load(open("labels.pkl","rb"))
    for i in xrange(1,3001):
        inp=pickle.load( open(r'./pickles/'+str(i)+'.pkl', 'rb'))
        inp1=np.zeros((10,1))
        inp1[a1[i]] = 1
        targets.append(inp1)
        inputs.append(inp)
    nn.train(inputs[:2100], targets[:2100], (inputs[2100:2550], targets[2100:2550]), 30,regularizer_type=None)
    c=0
    for j in xrange(0,450):
        a=pickle.load( open(r'./pickles/'+str(2551+j)+'.pkl', 'rb'))
        s=nn.predict(a)
        s1=a1[2551+j]
        if s == s1:
            c+=1
    #print j
    print "Correctly identified test cases : ",c
    print "Accuracy : ",((c*1.0)/(j+1))*100

else:
    nn=ann_sig.NeuralNetwork(16*8*3,8,10)
    inputs=[]
    targets=[]
    a1=pickle.load(open("labels.pkl","rb"))
    for i in xrange(1,3001):
        inp=pickle.load( open(r'./pickles/'+str(i)+'.pkl', 'rb'))
        inp1=np.zeros((10,1))
        inp1[a1[i]] = 1
        targets.append(inp1)
        inputs.append(inp)
    nn.train(inputs[:2100], targets[:2100], (inputs[2100:2550], targets[2100:2550]), 30,regularizer_type=None)
    c=0
    for j in xrange(0,450):
        a=pickle.load( open(r'./pickles/'+str(2551+j)+'.pkl', 'rb'))
        s=nn.predict(a)
        s1=a1[2551+j]
        if s == s1:
            c+=1
    #print j
    print "Correctly identified test cases : ",c
    print "Accuracy : ",((c*1.0)/(j+1))*100

