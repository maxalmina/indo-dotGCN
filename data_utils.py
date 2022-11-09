# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import random_split
from collections import Counter
import os.path

INFINITY_NUMBER = 1e12

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue
    return word_vec

def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = 'data_embedding/{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class Tokenizer4Bert:
    def __init__(self, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)

    def text_to_sequence(self, text, reverse=False):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return sequence

class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text
    
    def __build_stop_words__(fname, rank=50):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        stop_words = Counter()

        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            text_raw = text_left + " " + aspect + " " + text_right
            stop_words.update(text_raw.split())
        
        stop_words = stop_words.most_common(rank)
        
        return stop_words
    
    def __build_label_names__(fname, rank=10):
        
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        
        sentiment_dict_file = '/home/zeeeyang/2.researchs/aspect_sentiment/annotation_data/MAMS-for-ABSA/data/sentiment_dict.json'
        import json

        with open(sentiment_dict_file, 'r') as myfile:
            data=myfile.read()

        # parse file
        sent_dict = json.loads(data)
        def list_to_dict(word_list):
            word_set = set([word.lower() for word in word_list])
            return word_set 
            
        pos_words, neg_words, neu_words = list_to_dict(sent_dict['positive']), list_to_dict(sent_dict['negative']), list_to_dict(sent_dict['neutral'])
        pos_counter, neg_counter, neu_counter = Counter(), Counter(), Counter()
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            text_raw = text_left + " " + aspect + " " + text_right
            words = text_raw.split()
            
            pos_counter.update([ x for x in words if x in pos_words])
            neg_counter.update([ x for x in words if x in neg_words])
            neu_counter.update([ x for x in words if x in neu_words])
        
        pos_words = pos_counter.most_common(rank)
        neg_words = neg_counter.most_common(rank)
        neu_words = neu_counter.most_common(rank)
        
        polarity2label = {0: " ".join([x[0] for x in neg_words]), 1: " ".join([x[0] for x in neu_words]), 2: " ".join([x[0] for x in pos_words])}
        print(polarity2label)
        
        return polarity2label

    @staticmethod
    def __read_data__(fname, tokenizer, pos_tokenizer, rel_tokenizer, bert_tokenizer, polarity2label=None):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()
        print(len(idx2gragh))
        pos_lines = open(fname + ".pos", 'r').readlines()
        print(len(pos_lines))
        rel_lines = open(fname + ".rel", 'r').readlines()
        sent_dict = {}
        
        if os.path.isfile(fname+'.dist'):
            fin = open(fname+'.dist', 'rb')
            idx2dist = pickle.load(fin)
            fin.close()
        else:
            idx2dist = {}
            
        all_data = []
        
        if polarity2label is None:
            polarity2label = {0: 'disgusting horrible terrible lousy worst boring unhelpful ridiculous awful bad', \
                              1: 'so-so neutral average unbiased mixed alright', 
                              2: 'excellent wonderful fantastic perfect gem terrific spectacular outstanding exceptional delightful great good'}

        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            assert len(aspect) > 0 
            polarity = lines[i + 2].strip()

            original_line = text_left + " " + aspect + " " + text_right
            words = original_line.split()
            if original_line not in sent_dict:
                sent_id = len(sent_dict)
                sent_dict[original_line] = sent_id 
            else:
                sent_id = sent_dict[original_line]
                
            text_indices = tokenizer.text_to_sequence(original_line)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            assert len(aspect_indices) > 0
            
            left_indices = tokenizer.text_to_sequence(text_left)
            polarity = int(polarity)+1
            dependency_graph = idx2gragh[i]
            text_bert_indices = bert_tokenizer.text_to_sequence('[CLS] ' + original_line + ' [SEP] ' + aspect + " [SEP]")
            left_bert_indices = bert_tokenizer.text_to_sequence('[CLS] ' + text_left)
            
            labeled_bert_indices = bert_tokenizer.text_to_sequence('[CLS] ' + original_line + ' [SEP] ' + aspect +" " + polarity2label[polarity]+ " [SEP]")
            
            span_start = len(left_indices)
            span_end = len(left_indices) + len(aspect_indices)-1
            
            assert span_end < len(text_indices)

            word_lens = [ len(bert_tokenizer.text_to_sequence(word)) for word in original_line.split()]


            text_raw_bert_indices = bert_tokenizer.text_to_sequence("[CLS] " + original_line + " [SEP]")

            aspect_bert_indices = bert_tokenizer.text_to_sequence(aspect)
            labeled_aspect_bert_indices = bert_tokenizer.text_to_sequence(aspect + " " + polarity2label[polarity])

            aspect_bert_len = len(aspect_bert_indices)
            labeled_aspect_bert_len = len(labeled_aspect_bert_indices)

            bert_segments_ids = [0] * len(text_raw_bert_indices)  + [1] * (aspect_bert_len + 1)
            bert_token_masks = [1] * len(text_bert_indices)
            
            labeled_bert_segments_ids = [0] * len(text_raw_bert_indices)  + [1] * (labeled_aspect_bert_len + 1)
            labeled_bert_token_masks = [1] * len(labeled_bert_indices)
            assert len(bert_segments_ids) == len(text_bert_indices)

            pos_indices = pos_tokenizer.text_to_sequence(pos_lines[i//3].strip())
            
            if len(pos_indices) != len(text_indices):
                print("pos_len:" + str(len(pos_indices)))
                print(pos_indices)
                print("text_len:" + str(len(text_indices)))
                print(text_indices)
                sys.exit(0)
                
            assert(len(pos_indices) == len(text_indices))    
            rel_indices = rel_tokenizer.text_to_sequence(rel_lines[i//3].strip())
            target = (len(left_indices), len(left_indices)+len(aspect_indices))
            
            if i in idx2dist: #cache 
                dist_to_target = idx2dist[i]
            else:
                dist_to_target = get_dist_to_target(dependency_graph, target, [-1]*len(dependency_graph))
                idx2dist[i] = dist_to_target

            data = {
                'sent_id': sent_id, 
                'span': (span_start, span_end),
                'text': text_left + " $" + aspect + "$ " + text_right, 
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'aspect_bert_indices': aspect_bert_indices, 
                'left_indices': left_indices,
                'left_bert_indices': left_bert_indices, 
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'pos_indices': pos_indices,
                'rel_indices': rel_indices,
                'text_bert_indices': text_bert_indices,
                'text_raw_bert_indices': text_raw_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'bert_token_masks': bert_token_masks,
                'labeled_bert_segments_ids': labeled_bert_segments_ids,
                'labeled_bert_token_masks': labeled_bert_token_masks,
                'word_lens': word_lens,
                'words': words,
                'labeled_bert_indices': labeled_bert_indices,
                'dist_to_target': dist_to_target
            }

            all_data.append(data)

        if not os.path.isfile(fname+'.dist'):
            with open(fname+'.dist', 'wb') as file:
                pickle.dump(idx2dist, file)
        
        return all_data
    
    def __init__(self, dataset='twitter', embed_dim=300, bert_name='bert-base-uncased', valset_ratio=None):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            
            'mams': {
                'train': './datasets/mams/mams_train.raw', 
                'dev': './datasets/mams/mams_val.raw', 
                'test': './datasets/mams/mams_test.raw'
            },
            'small': {
                'train': './datasets/mams_small/mams_train.raw', 
                'dev': './datasets/mams_small/mams_val.raw', 
                'test': './datasets/mams_small/mams_test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'laptop14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'twitter': {
                'train': './datasets/Z_data/train.raw', 
                'dev': './datasets/Z_data/dev.raw', 
                'test': './datasets/Z_data/test.raw'
            }
        }

        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        polarity2label = None 

        folder_token = 'data_token/'
        
        if os.path.exists(folder_token+dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(folder_token+dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(folder_token+dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        
        if os.path.exists(folder_token+dataset+'_pos2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(folder_token+dataset+'_pos2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 pos_tokenizer = Tokenizer(word2idx=word2idx)
        else:
            pos_tokenizer = Tokenizer()
            pos_text = open(fname[dataset]['train']+".pos", "r").readlines()
            pos = []
            for line in pos_text:
                items = line.strip().split()
                pos.extend(items)
            pos_tokenizer.fit_on_text(" ".join(pos))
            with open(folder_token+dataset+'_pos2idx.pkl', 'wb') as f:
                 pickle.dump(pos_tokenizer.word2idx, f)
                 
        if os.path.exists(folder_token+dataset+'_rel2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(folder_token+dataset+'_rel2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 rel_tokenizer = Tokenizer(word2idx=word2idx)
        else:
            rel_tokenizer = Tokenizer()
            rel_text = open(fname[dataset]['train']+".rel", "r").readlines()
            rel = []
            for line in rel_text:
                items = line.strip().split()
                rel.extend(items)
            rel_tokenizer.fit_on_text(" ".join(rel))
            with open(folder_token+dataset+'_rel2idx.pkl', 'wb') as f:
                 pickle.dump(rel_tokenizer.word2idx, f)
        
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        bert_tokenizer = Tokenizer4Bert(bert_name)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer, pos_tokenizer, rel_tokenizer, bert_tokenizer, polarity2label))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer, pos_tokenizer, rel_tokenizer, bert_tokenizer, polarity2label))
        if len(fname[dataset]) ==  3: 
            self.dev_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['dev'], tokenizer, pos_tokenizer, rel_tokenizer, bert_tokenizer, polarity2label))
        else: 
            self.dev_data = self.test_data 
        
        if valset_ratio is not None:     
            assert 0 <= valset_ratio < 1
            if valset_ratio > 0:
                valset_len = int(len(self.train_data) * valset_ratio)
                self.train_data, self.dev_data = random_split(self.train_data, (len(self.train_data)-valset_len, valset_len))
             
            
        print("[tlog] train num: " + str(len(self.train_data)))
        
        if len(fname[dataset]) ==  3: 
            print("[tlog] dev num: " + str(len(self.dev_data)))
        print("[tlog] test num: " + str(len(self.test_data)))

        self.tokenizer = tokenizer
        self.pos_tokenizer = pos_tokenizer
        self.rel_tokenizer = rel_tokenizer
        self.bert_tokenizer = bert_tokenizer

def get_dist(i, target, adj, seen):
    seen.append(i)
    if i in target:
        return 0
    else:
        children = []
        for j in range(len(adj)):
            if adj[i][j] == 1 and i != j and j not in seen:
                children.append(j)
        md = INFINITY_NUMBER
        for c in children:
            d = get_dist(c, target, adj, seen) + 1
            if d < md:
                md = d
        return md

def get_dist_to_target(adj, target, dist):
    target = list(range(target[0], target[1]))
    for i in range(len(adj)):
        dist[i] = get_dist(i, target, adj, [])
    assert all(d != -1 for d in dist), dist
    return [d+1 for d in dist]
