## Code for Discrete Opinion Tree Induction for Aspect Sentiment Analysis 

### 1. Download Glove 
Put glove embeddings `glove.840B.300d.txt` in the folder `glove`. This is only used as a placeholder, we do not use the glove embeddings in this project. 

### 2. Run Mams 

Use the default settings for `dotGCN`
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset mams --model rlgcn
```
or use a different random seed 

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset mams --model rlgcn --seed 2333 
```
To use other datasets, you can change the dataset name `mama` to the desired one such as `rest14`, `lap14`, `twitter`, `rest15` and `rest16`. 

### 3. Run Baselines 
Run `BERT-SPC*` 
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset mams --model bert-spc
```
Run `depGCN+sd+BERT`
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset mams --model depgcnv2
```
Run `viGCN`
```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset mams --model vaerlgcn --att_weight 0.01
```