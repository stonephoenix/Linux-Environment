#!/usr/bin/python
import sys
import re

def makeDict(filename):
    dict = {}
    dict['</s>'] = 1
    dict['<s>'] = 2
    id = 3
    with open(filename) as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip(' \r\n')
            lines[i] = re.split(' |\t', lines[i])[i]
            if lines[i] not in dict.keys():
                dict[lines[i]] = id
                id = id + 1
    return dict

class trieNode:
    def __init__(self, wordid, logweight, logbow):
        self.wordid = wordid
        self.logweight = logweight
        self.logbow = logbow
        self.children = {}
        
    def add_child(self, child):
    ''' Does NOT check the existence of input child in self.children'''
        if child.wordid is None:
            raise AttributeError('child.wordid should NOT be None')
        self.children[child.wordid] = child
        return self
        
class trieTree:

    def __init__(self, dict):
        self.dict = dict
        self.head = trieNode()
        self.count = []
    
    def add_branch(ngram, logweight, logbow):
    ''' Note: lower ngram should be processed in advance 
    ngram: list of wordid
    '''
        node = trieNode(ngram[-1], logweight, logbow)
        head = self.head
        while i in range(len(ngram)-1):
            head = head.children[ngram[i]]
        self.head.add_child(node)
        if len(ngram) <= len(self.count):
            self.count[len(ngram)-1] = self.count[len(ngram)-1] + 1
        elif len(ngrram) == len(self.count) + 1:
            self.count.append(1)
        else:
            raise AttributeError("ngram item should be added in order")
        
    def add_line(line, dict):
        line = line.rstrip(' \r\n')
        section = line.split('\t')
        if len(section) < 2:
            return
        logweight = float(section[0])
        logbow = None
        if len(section) == 3:
            logbow = float(section[2])
        words = section[1].split(' ')
        ngram = []
        for w in words:
            ngram.append(dict[w])
        self.add_branch(ngram, logweight, logbow)
        
    def __str__(self):
        dict_re = {}
        for word in self.dict.keys():
            dict_re[self.dict[word]] = word
        print '\data\\'
        for i in range(len(self.count)):
            print 'ngram ' + str(i+1) + '=' + str(self.count[i])
        print ''
        nodes = [head.children[key] for key in head.children.keys()]        
        while len(nodes) > 0:
            print '\\' + str(i+1) + '-gram:'
            n = len(nodes)
            while n > 0:
                node = nodes.pop(0)
                if node.logbow is None:
                    print str(node.logweight) + '\t' + dict_re[node.wordid]
                else:
                    print str(node.logweight) + '\t' + dict_re[node.wordid] + '\t' + str(node.logbow)
                n = n - 1
                if node.children is not None:
                    for wordid in node.children.keys():
                        nodes.append(node.children[wordid])
            print ''
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage: sys.argv[0] wordlist arprfile'
        exit()
    wordlist = argv[1]
    arpafile = argv[2]
    
    dict = makeDict(wordlist)
    tree = trieTree()
    with open(arpafile) as f:
        line = f.readline()
        
