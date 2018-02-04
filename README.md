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

```bash
python main.py --gpu
```
