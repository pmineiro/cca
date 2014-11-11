from sys import argv,stdin
from bisect import bisect
import gzip

def dist(x,y):
    return sum((xi-yi)**2 for xi,yi in zip(x,y))

def nn(q,d):
    dists=[(dist(d[w],q),w) for w in d]
    dists.sort()
    return dists[:5]

f=gzip.open(argv[1], 'rb')
k=int(argv[2])
eigendict=dict()
for line in f:
    parts=line.strip().split()
    word=parts[0]
    eigendict[word]=map(float,parts[1:k])
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
        query=[z+y-x for (x,y,z) in zip(eigendict[w1],eigendict[w2],eigendict[w3])]
        if w1 == w2 and w2 == w3:
            print 'nearest neighbors of %s are ...'%(w1)
        else:
            print '%s is to %s as %s is to ...'%(w1,w2,w3)
        print nn(query,eigendict)
        line=raw_input()

if __name__=='__main__':
    main()
