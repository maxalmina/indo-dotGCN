# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import stanfordnlp
import stanza
import sys

#nlp = spacy.load('en_core_web_sm')

'''
config = {
        'processors': 'tokenize,pos,depparse',
        'tokenize_pretokenized': True,
}
nlp = stanfordnlp.Pipeline(**config)
'''

'''spacy'''

nlp = stanza.Pipeline('en', tokenize_pretokenized=True) # initialize English neural pipeline

def dependency_adj_matrix(text):
    #print(text)
    document = nlp(text)
    s_len = 0
    for sentence in document.sentences:
        s_len += len(sentence.words)
    #sys.exit(0)
    seq_len = len(text.split())
    assert s_len == seq_len
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    pos = []
    dep_rel = []
    i = 0 
    for sentence in document.sentences:
        #print("[tlog] sentence: " + str(sentence))
        for word in sentence.words: 
        #print("[tlog] token: " + str(token.pos_))
            word_index = int(word.id) -1 
            assert(word_index >= 0)
            if word_index + i  < seq_len:  #there are some bugs for here such as SPACE
                pos.append(word.pos)
                dep_rel.append(word.deprel)
        #print("[tlog] token: " + str(token.dep_))
        #print("[tlog] token.i: " + str(token.i))
        #print("[tlog] token.children: " + str([child for child in token.children]))
        #print("\n")
        #sys.exit(0) governor
            if word_index + i < seq_len:
                index = word_index + i 
                matrix[index][index] = 1
                if int(word.head)>0:
                    head_index = int(word.head) -1 + i 
                    #assert(head_index >= 0)
                    if head_index >= 0: 
                        matrix[head_index][index] = 1
                        matrix[index][head_index] = 1
                
        i += len(sentence.words)           
    #print("[tlog] matrix: " + str(matrix))
    #sys.exit(0)
    return matrix, pos, dep_rel

def dependency_adj_matrix2(text):
    # https://spacy.io/docs/usage/processing-text
    #print("[tlog] text: " + str(text)) # Maybe for parsing, we should not lower case this 
    document = nlp(text)
    #print("[tlog] document: " + str(document))
    #sys.exit(0)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    pos = []
    dep_rel = []
    for token in document:
        #print("[tlog] token: " + str(token))
        #print("[tlog] token: " + str(token.pos_))
        if token.i < seq_len:  #there are some bugs for here such as SPACE
            pos.append(token.tag_)
            dep_rel.append(token.dep_)
        #print("[tlog] token: " + str(token.dep_))
        #print("[tlog] token.i: " + str(token.i))
        #print("[tlog] token.children: " + str([child for child in token.children]))
        #print("\n")
        #sys.exit(0)
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children: # tzy: do not distinguish the arc types 
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
                    
    #print("[tlog] matrix: " + str(matrix))
    #sys.exit(0)
    return matrix, pos, dep_rel

def dependency_adj_matrix3(text):
    # https://spacy.io/docs/usage/processing-text
    #print("[tlog] text: " + str(text)) # Maybe for parsing, we should not lower case this 
    document = nlp(text)
    #print("[tlog] document: " + str(document))
    #sys.exit(0)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    pos = []
    dep_rel = []
    for token in document:
        #print("[tlog] token: " + str(token))
        #print("[tlog] token: " + str(token.pos_))
        if token.i < seq_len:  #there are some bugs for here such as SPACE
            pos.append(token.tag_)
            dep_rel.append(token.dep_)
        #print("[tlog] token: " + str(token.dep_))
        #print("[tlog] token.i: " + str(token.i))
        #print("[tlog] token.children: " + str([child for child in token.children]))
        #print("\n")
        #sys.exit(0)
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children: # tzy: do not distinguish the arc types 
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
                    
    #print("[tlog] matrix: " + str(matrix))
    #sys.exit(0)
    return matrix, pos, dep_rel

def process(filename):
    print(filename)
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    pos_out = open(filename+'.pos', 'w')
    rel_out = open(filename+'.rel', 'w')
    line_count = 0
    for i in range(0, len(lines), 3):
        #text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        #aspect = lines[i + 1].lower().strip()
        text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].strip()
        line_count += 1
        if line_count % 1000 == 0:
            print(line_count)
        adj_matrix, pos, rel = dependency_adj_matrix(text_left.strip()+' '+aspect+' '+text_right.strip())
        #adj_matrix, pos, rel = dependency_adj_matrix2(text_left.strip()+' '+aspect+' '+text_right.strip())
        #adj_matrix, pos, rel = dependency_adj_matrix3(text_left.strip()+' '+aspect+' '+text_right.strip())
        idx2graph[i] = adj_matrix
        pos_out.write(" ".join(pos)+"\n")
        rel_out.write(" ".join(rel)+"\n")
    pickle.dump(idx2graph, fout)        
    fout.close() 

if __name__ == '__main__':
    #process('./datasets/mams_small/mams_train.raw')
    #process('./datasets/mams_small/mams_val.raw')
    #process('./datasets/mams_small/mams_test.raw')
    process(sys.argv[1])
    
    sys.exit(0)
    #process('./datasets/merge/train.raw')
    process('./datasets/merge/test.raw')
    sys.exit(0)
    #process('./datasets/semeval14/restaurant_train.raw')
    #process('./datasets/semeval14/restaurant_test.raw')
    #'''
    #process('./datasets/semeval16/restaurant_train.raw')
    #process('./datasets/semeval16/restaurant_test.raw')
    sys.exit(0)
    #process('./datasets/semeval14/restaurant_train1.raw')
    #process('./datasets/semeval14/restaurant_test1.raw')
    #sys.exit(0)
    process('./datasets/mams/mams_train.raw')
    process('./datasets/mams/mams_val.raw')
    process('./datasets/mams/mams_test.raw')
    sys.exit(0)
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    #sys.exit(0)
    #process('./datasets/semeval14/restaurant_train.raw')
    #process('./datasets/semeval14/restaurant_test.raw')

    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    #'''
    process('./datasets/T_data/train.raw')
    process('./datasets/T_data/test.raw')
    
    process('./datasets/Z_data/train.raw')
    process('./datasets/Z_data/dev.raw')
    process('./datasets/Z_data/test.raw')
    
