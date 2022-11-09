# -*- coding: utf-8 -*-
import os
import math
import argparse
import random

import numpy
from sklearn import metrics

import torch
import torch.nn as nn

from data_utils import ABSADatesetReader
from bucket_iterator import BucketIterator
from models import BertSPC, RLGCN, RLVAEGCN, DepGCNv2

from transformers import AdamW
from transformers.models.bert.modeling_bert import BertModel
from transformers.optimization import get_linear_schedule_with_warmup

from allennlp.modules.scalar_mix import ScalarMix

class Instructor:
    def __init__(self, opt):
        self.opt = opt

        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim) #valset_ratio=0.1 if opt.dataset != "mams" else None
        
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
        self.dev_data_loader = BucketIterator(data=absa_dataset.dev_data, batch_size=opt.batch_size, shuffle=False)
        opt.pos_size = len(absa_dataset.pos_tokenizer.word2idx)
        opt.rel_size = len(absa_dataset.rel_tokenizer.word2idx)
        opt.pos_dim = 30
        opt.rel_dim = 30
        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.
        self.train_examples = len(absa_dataset.train_data)
        self.use_dice_loss = False 
        
        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
      
    def _reset_params(self):
         for child in self.model.children():
            if type(child) not in [BertModel, ScalarMix, GRUCell,  nn.Embedding]:  # skip bert params
                for name, p in child.named_parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    
    def lr_exponential_decay(self, optimizer, decay_rate):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
            print(f"Learning rate is setted as: {param_group['lr']}")

    def load_model(self):
        print('loading model {0} ...'.format(opt.load_state_dict_path))
        self.model.load_state_dict(torch.load(opt.load_state_dict_path))
        print("[tlog] load success")

    def _train(self, criterion, optimizer, scheduler=None):
        max_dev_acc = 0
        max_dev_f1 = 0
        
        max_dev_acc = 0
        max_dev_f1 = 0
        
        global_step = 0
        continue_not_increase = 0
        save_index = 0
        
        if not self.opt.use_single_optimizer: 
            bert_optimizer, non_bert_optimizer = optimizer
        
            bert_params = []
            non_bert_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if "bert_model" in name: 
                        bert_params.append(param)
                    else:
                        non_bert_params.append(param)
        else:
            bert_params = [ param for name, param in self.model.named_parameters() if param.requires_grad]

        #max_temp = 1.0 
        #min_temp = 0.2 

        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            increase_flag = False
            
            if not self.opt.use_single_optimizer: 
                if epoch > 0: 
                    self.lr_exponential_decay(non_bert_optimizer, 0.97)
            
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                temperature = None
                global_step += 1
                
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                if bert_optimizer is not None: 
                    bert_optimizer.zero_grad()
                    
                non_bert_optimizer.zero_grad()
                
                inputs = [sample_batched[col].to(self.opt.device) if (col !="word_lens" and col != "words") else sample_batched[col] for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                
                outputs, loss = self.model(inputs, temperature=temperature, labels=targets)
                
                loss.backward()
                
                if bert_optimizer is not None: 
                    torch.nn.utils.clip_grad_norm_(parameters=bert_params, max_norm=5.0)
                
                    bert_optimizer.step()
                
                if not self.opt.use_single_optimizer: 
                    torch.nn.utils.clip_grad_norm_(parameters=non_bert_params, max_norm=5.0)
                    non_bert_optimizer.step()
                
                if scheduler is not None: 
                    scheduler.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    dev_acc, dev_f1 = self._evaluate_acc_f1(self.dev_data_loader)
                    
                    use_max_acc = True 
                    if use_max_acc: 
                        a, b = dev_acc, dev_f1 
                        dev_acc, dev_f1 = b, a 
                    
                    if id(self.dev_data_loader) == id(self.test_data_loader):
                        test_acc = dev_acc 
                        test_f1 = dev_f1 
                    else: 
                        test_acc, test_f1 = self._evaluate_acc_f1(self.test_data_loader)
                    
                    if dev_acc > max_dev_acc: #tzy: not at the same time 
                        max_dev_acc = dev_acc
                        max_test_acc = test_acc 
                    if dev_f1 > max_dev_f1:
                        increase_flag = True
                        max_dev_f1 = dev_f1
                        max_test_f1 = test_f1
                        if dev_f1 > self.global_f1:
                            self.global_f1 = dev_f1
                            print('>>> best model saved.')
                            if self.opt.save:
                                save_index = (save_index + 1)%5
                                save_index = 0
                                path = 'saved_models/state_dict_'+self.opt.model_name+'/'+self.opt.model_name+'_'+self.opt.dataset + "." + str(save_index)+'.pkl'
                                print(path)
                                torch.save(self.model.state_dict(), path)
                            
                    print('loss: {:.4f}, acc: {:.4f}, dev_acc: {:.5f}, dev_f1: {:.5f}, tst_acc: {:.5f}, tst_f1: {:.5f}'.format(loss.item(), train_acc, dev_acc, dev_f1, test_acc, test_f1))
            if increase_flag == False:
                continue_not_increase += 1
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0    
        return max_dev_acc, max_dev_f1, max_test_acc, max_test_f1 

    def _evaluate_acc_f1(self, test_data_loader):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) if (col !="word_lens" and col != "words") else t_sample_batched[col] for col in self.opt.inputs_cols] 
                t_targets = t_sample_batched['polarity'].to(opt.device)
                
                
                t_outputs = self.model(t_inputs)
                

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def run(self, repeats=1):
        # Loss and Optimizer
        if self.use_dice_loss: 
            criterion = MulticlassDiceLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        if not self.opt.use_single_optimizer:
            bert_params = []
            non_bert_params = []
            for name, param in self.model.named_parameters():
                if param.requires_grad: 
                    print(name)
                    if "bert_model" in name: 
                        bert_params.append(param)
                    else:
                        non_bert_params.append(param)
                    
        total_train_steps = int(self.train_examples / self.opt.batch_size * self.opt.num_epoch)
         
        if self.opt.use_bert_adam: 
            from models.optimizer import BertAdam
            if not self.opt.use_single_optimizer:
                if len(bert_params) > 0: 
                    bert_optimizer = BertAdam(bert_params,
                                    lr=self.opt.learning_rate,
                                    warmup=self.opt.warmup_proportion,
                                    t_total=total_train_steps)
                else:
                    bert_optimizer = None

                non_bert_optimizer = torch.optim.Adam(non_bert_params, lr=self.opt.learning_rate, weight_decay=1e-4)
            else:
                params = [ param for name, param in self.model.named_parameters() if param.requires_grad]
                
                bert_optimizer = BertAdam(params,
                                lr=self.opt.learning_rate,
                                warmup=self.opt.warmup_proportion,
                                t_total=total_train_steps)
        else: 
            if not self.opt.use_single_optimizer:
                bert_optimizer = self.opt.optimizer(bert_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
                non_bert_optimizer = self.opt.optimizer(non_bert_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            else:
                params = [ param for name, param in self.model.named_parameters() if param.requires_grad]
                bert_optimizer = self.opt.optimizer(params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        
        if self.opt.use_bert_adam and bert_optimizer is not None: 
            scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=int(total_train_steps * self.opt.warmup_proportion), num_training_steps=total_train_steps)
        else: 
            scheduler = None 
        
        
        if not os.path.exists('log/'):
            os.mkdir('log/')

        f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')

        max_test_acc_avg = 0
        max_test_f1_avg = 0
        
        max_dev_acc_avg = 0
        max_dev_f1_avg = 0
        
        assert(repeats == 1)
        
        for i in range(repeats):
            print('repeat: ', (i+1))
            f_out.write('repeat: '+str(i+1))
            self._reset_params()

            max_dev_acc, max_dev_f1, max_test_acc, max_test_f1 = self._train(criterion, [bert_optimizer, non_bert_optimizer] if not self.opt.use_single_optimizer else [bert_optimizer], scheduler)
            print('max_dev_acc: {0}     max_dev_f1: {1}'.format(max_dev_acc, max_dev_f1))
            print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            
            f_out.write('max_dev_acc: {0}, max_dev_f1: {1}\n'.format(max_dev_acc, max_dev_f1))
            f_out.write('max_test_acc: {0}, max_test_f1: {1}\n'.format(max_test_acc, max_test_f1))
            
            max_dev_acc_avg += max_dev_acc
            max_dev_f1_avg += max_dev_f1
            
            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            print('#' * 100)
        
        print("max_dev_acc_avg:", max_dev_acc_avg / repeats)
        print("max_dev_f1_avg:", max_dev_f1_avg / repeats)
        
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)

        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lstm', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, rest14, laptop14, rest15, rest16, t, z, mams')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int) #300
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--use_aux_aspect', default=False, type=bool)
    parser.add_argument('--use_single_bert', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int) #776
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--lambda_p', default=0.8, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--heads', default=2, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--sublayer_first', default=1, type=int)
    parser.add_argument('--sublayer_second', default=1, type=int)
    parser.add_argument('--gcn_dropout', default=0.3, type=int)
    parser.add_argument('--dropout', default=0.0, type=int)
    #RL
    parser.add_argument('--rl_weight', default=0.1, type=float)
    parser.add_argument('--sample-num', default=3, type=int, help='sample num for reinforce')
    parser.add_argument('--clf-num-layers', default=1, type=int)
    parser.add_argument('--load_state_dict_path', default=None, type=str)
    #rl loss weight 
    parser.add_argument('--td_weight', default=0.1, type=float) #tree distance regularzied weight
    parser.add_argument('--ent_weight', default=0.0001, type=float) #entropy weight 
    parser.add_argument('--att_weight', default=0.1, type=float) # attention weight

    opt = parser.parse_args()
    print("[tlog] opt: " + str(opt))

    model_classes = {
        'depgcn2': DepGCNv2,
        'rlgcn': RLGCN,
        'bert-spc': BertSPC,
    }
    input_colses = {
        'bert-spc': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph', 'pos_indices', 'rel_indices', 'text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'bert_token_masks', 'word_lens'],
        'rlgcn': ['text_indices', 'aspect_indices', 'aspect_bert_indices', 'left_indices', 'left_bert_indices', 'dependency_graph', 'pos_indices', 'rel_indices', 'text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'bert_token_masks', 'word_lens', 'words', 'aux_aspect_targets'],
        'depgcn2': ['text_indices', 'aspect_indices', 'aspect_bert_indices', 'left_indices', 'left_bert_indices', 'dependency_graph', 'pos_indices', 'rel_indices', 'text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'bert_token_masks', 'word_lens', 'words', 'aux_aspect_targets', 'dist_to_target'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamw': AdamW, 
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    opt.use_bert_adam = True 
    opt.use_single_optimizer = False 

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run(repeats=1)
