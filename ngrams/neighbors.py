from sys import argv, stdin
import numpy as np, h5py, heapq

epsilon=0.001

f=h5py.File(argv[1],'r');
data=f.get('megaproj');
data=np.array(data);
words=open(argv[2]);
eigendict=dict()
reverseeigendict=list()
for line in words:
    parts=line.strip().split()
    word=parts[1]
    eigendict[word]=int(parts[0])-1
    reverseeigendict.append(word)
f.close()
words.close()

def main():
    print 'ready ...'
    line=raw_input()
    while line:
        w1,w2,w3=line.strip().split()
        for wi in w1,w2,w3:
            if wi not in eigendict:
                print "%s not found"%wi
                break
        else: #no break
            #query=data[:,eigendict[w3]]+data[:,eigendict[w2]]-data[:,eigendict[w1]];
            #rawscores=zip(query.T.dot(data),range(len(eigendict)))
            #scores=heapq.nlargest(5, rawscores)

            rawscoresone=data[:,eigendict[w3]].T.dot(data);
            rawscorestwo=data[:,eigendict[w2]].T.dot(data);
            rawscoresthree=data[:,eigendict[w1]].T.dot(data);
            cos3mul=zip(np.log(0.5+0.5*rawscoresone)+np.log(0.5+0.5*rawscorestwo)-np.log(epsilon+0.5+0.5*rawscoresthree),range(len(eigendict)))
            scores=heapq.nlargest(5,cos3mul)

            if w1 == w2 == w3:
                print 'nearest neighbors of %s are ...'%(w1)
            else:
                print '%s is to %s as %s is to ...'%(w1,w2,w3)
            print [(x,reverseeigendict[y]) for (x,y) in scores[:5]]
        line=raw_input()

if __name__=='__main__':
    main()
