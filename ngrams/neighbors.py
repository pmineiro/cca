from sys import argv,stdin
from bisect import bisect
import numpy as np, h5py

def norm(x):
    return np.inner(x,x)

def dist(x,y):
    return norm(x-y)

def nn(q,d,v):
    dists=[(dist(v[:,d[w]],q),w) for w in d]
    dists.sort()
    return dists[:5]

f=h5py.File(argv[1],'r');
data=f.get('megaproj');
data=np.array(data);
words=open(argv[2],'r');
k=int(argv[3])
eigendict=dict()
for line in words:
    parts=line.strip().split()
    word=parts[1]
    eigendict[word]=int(parts[0])-1
f.close()

vocab=eigendict.keys()
vocab.sort()

def main():
    print 'ready ...'
    line=raw_input()
    while line:
        w1,w2,w3=line.strip().split()
        if w1 not in eigendict:
            print "%s not found"%w1
            continue
        if w2 not in eigendict:
            print "%s not found"%w2
            continue
        if w3 not in eigendict:
            print "%s not found"%w3
            continue
        query=data[:,eigendict[w3]]+data[:,eigendict[w2]]-data[:,eigendict[w1]];
        if w1 == w2 and w2 == w3:
            print 'nearest neighbors of %s are ...'%(w1)
        else:
            print '%s is to %s as %s is to ...'%(w1,w2,w3)
        print nn(query,eigendict,data)
        line=raw_input()

if __name__=='__main__':
    main()
