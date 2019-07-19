# An Implementation of Deep Exhaustive Model for Nested NER

Original paper: [Sohrab, M. G., & Miwa, M. (2018). Deep Exhaustive Model for Nested Named Entity Recognition. In 2018 EMNLP](http://aclweb.org/anthology/D18-1309)

# Requirements
* `python        3.6.7`
* `pytorch       1.0.0`
* `numpy         1.15.3`
* `gensim        3.6.0`
* `scikit-learn  0.20.0`
* `joblib        0.12.5`

# Data Format
Our processed `GENIA` dataset is in `./data/`.

The data format is the same as in [Neural Layered Model, Ju et al. 2018 NAACL](https://github.com/meizhiju/layered-bilstm-crf) 
>Each line has multiple columns separated by a tab key. 
>Each line contains
>```
>word	label1	label2	label3	...	labelN
>```
>The number of labels (`N`) for each word is determined by the maximum nested level in the data set. `N=maximum nested level + 1`
>Each sentence is separated by an empty line.
>For example, for these two sentences, `John killed Mary's husband. He was arrested last night` , they contain four entities: John (`PER`), Mary(`PER`), Mary's husband(`PER`),He (`PER`).
>The format for these two sentences is listed as following:
>```
>John    B-PER   O   O
>killed  O   O   O
>Mary    B-PER   B-PER   O
>'s  O   I-PER   O
>husband O   I-PER   O
>.   O   O   O
>
>He    B-PER   O   O
>was  O   O   O
>arrested  O   O   O
>last  O   O   O
>night  O   O   O
>.  O   O   O
>```

# Pre-trained word embeddings
* [Pre-trained word embeddings](https://drive.google.com/open?id=0BzMCqpcgEJgiUWs0ZnU0NlFTam8) used here is the same as in [Neural Layered Model](https://github.com/meizhiju/layered-bilstm-crf) 

# Setup
Download pre-trained embedding above, unzip it, and place `PubMed-shuffle-win-30.bin` into `./data/embedding/`

# Usage
## Training

```sh
python3 train.py
```
trained best model will be saved at `./data/model/`
## Testing
 set `model_url` to the url of saved model in training in `main()` of `eval.py`
```sh
python3 eval.py
```
