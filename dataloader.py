import pandas as pd
import random

# def read_corpus(path, clean=True, MR=True, encoding='utf8', shuffle=False, lower=True):
#     data = []
#     labels = []
#     with open(path, encoding=encoding) as fin:
#         for line in fin:
#             if MR:
#                 label, sep, text = line.partition(' ')
#                 label = int(label)
#             else:
#                 label, sep, text = line.partition(',')
#                 label = int(label) - 1
#             if clean:
#                 text = clean_str(text.strip()) if clean else text.strip()
#             if lower:
#                 text = text.lower()
#             labels.append(label)
#             data.append(text.split())

#     if shuffle:
#         perm = list(range(len(data)))
#         random.shuffle(perm)
#         data = [data[i] for i in perm]
#         labels = [labels[i] for i in perm]

#     return data, labels

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def read_corpus(path, clean=False, encoding='utf8', shuffle=False, lower=True):
    # TODO fix splitting to split word and punctuation separate and remove lower to keep text as close to original as possible.
    df = pd.read_csv(path, encoding=encoding)
    
    # Assumes your CSV has 'label' and 'text' columns
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV must contain 'label' and 'text' columns")
    
    data = []
    labels = []

    for _, row in df.iterrows():
        text = row['text']
        label = int(row['label'])

        if clean:
            text = clean_str(text.strip()) # NO cleaning because all models trained without cleaning
        if lower:
            text = text.lower()

        labels.append(label)
        data.append(text.split())

    if shuffle:
        combined = list(zip(data, labels))
        random.shuffle(combined)
        data, labels = zip(*combined)
        data, labels = list(data), list(labels)

    return data, labels