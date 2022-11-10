## Code for Discrete Opinion Tree Induction for Aspect Sentiment Analysis 

### 1. New Folders
create 3 new folders `glove`, `data_embedding`, and `data_token`

### 2. Download Glove 
Download and put glove embeddings `glove.840B.300d.txt` in the new folder `glove`. This is only used as a placeholder, we do not use the glove embeddings in this project. 

### 3. Run Mams 

Use the default settings for `dotGCN`
```
python train.py --dataset mams --model rlgcn
```
or use a different random seed 

```
python train.py --dataset mams --model rlgcn --seed 2333 
```
To use other datasets, you can change the dataset name `mama` to the desired one such as `rest14`, `laptop14`, `twitter`, `small`. 

### 4. Run Baselines 
Run `BERT-SPC*` 
```
python train.py --dataset mams --model bert-spc
```
Run `depGCN+sd+BERT`
```
python train.py --dataset mams --model depgcnv2
```
Run `viGCN`
```
python train.py --dataset mams --model vaerlgcn --att_weight 0.01
```