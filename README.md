# sentence_classification

## Requirements

- Python 2.7
- Pytorch


## Data preparation

- Download Stanford CoreNLP and GloVe, then extract to `data/`.

    ```bash
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
    unzip stanford-corenlp-full-2017-06-09.zip
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip
    ```

### TREC

1. Download the dataset.

    ```bash
    wget http://cogcomp.org/Data/QA/QC/train_5500.label
    wget http://cogcomp.org/Data/QA/QC/TREC_10.label
    ```

2. Parse raw data.

    ```bash
    python preprocess.py
    ```

3. Get json data.

    ```bash
    ./gen_json.sh
    ```


## Train

- Without glove

```bash
python main.py --gpu
```

- With glove

```bash
python main.py --gpu --glove_path=data/glove.840B.300d.txt --lr=0.008 --lr_milestones=11
```


## Evaluate

- Without glove

```bash
python main.py --gpu --mode=eval \
    --checkpoint_path=models/TREC_test_batchsize25_input300_hidden100_lr0.001_seed10137_epoch48.pth
```

- With glove

```bash
python main.py --gpu --mode=eval \
    --checkpoint_path=models/TREC_test_batchsize25_input300_hidden100_lr0.008_ms11_wc0.0001_glove_seed10137_epoch12.pth
```
