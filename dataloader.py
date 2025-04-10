import pandas as pd
import random
import re

def tokenize_ukrainian(text):
    pattern = r"(\s+|[^\w\s']+|[\w']+)"
    tokens = re.findall(pattern, text)
    
    final_tokens = []
    i = 0
    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and tokens[i].isalpha()
            and tokens[i+1] == "'"
            and tokens[i+2].isalpha()
        ):
            final_tokens.append(tokens[i] + tokens[i+1] + tokens[i+2])
            i += 3
        else:
            final_tokens.append(tokens[i])
            i += 1
    return final_tokens


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


def read_corpus(path, clean=False, encoding='utf8', shuffle=False, lower=False):
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
        # data.append(text.split())
        data.append(tokenize_ukrainian(text))

    if shuffle:
        combined = list(zip(data, labels))
        random.shuffle(combined)
        data, labels = zip(*combined)
        data, labels = list(data), list(labels)

    return data, labels