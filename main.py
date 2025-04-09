import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

from dataloader import read_corpus
from sentence_sim import SBERT
import criteria
from prepare_synonym_dict import read_and_clean_synonym_dict
import pymorphy2
import re

morph = pymorphy2.MorphAnalyzer(lang='uk')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

path_to_ulif = '/home/mudryi/phd_projects/synonym_attack/synonyms_dictionaries/ulif_clean.json'
synonym_dict = read_and_clean_synonym_dict(path_to_ulif)

def get_correct_parsed_result(parsing_results, target_pos, target_gender=None):

    for result in parsing_results:
        if result.tag.POS == target_pos and target_gender and result.tag.gender == target_gender:
            return result

    for result in parsing_results:
        if result.tag.POS == target_pos:
            return result

    return None

def lower_grammar_restrictions(grammemes, target_gender):
    grammemes_to_remove = [str(target_gender), 'Refl']
    
    new_grammemes = set(item for item in list(grammemes) if item not in grammemes_to_remove)
    return new_grammemes

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

def is_word_token(tok):
    return tok.strip().isalpha()  # This excludes whitespace and punctuation
    
def get_normal_form(word, morph):
    parsed = morph.parse(word)
    if parsed:
        return parsed[0].normal_form
    else:
        return word
    
def get_all_synonyms(word):
    normal_form = get_normal_form(word, morph)

    if word in synonym_dict:
        return synonym_dict[word]
    elif normal_form in synonym_dict:
        return synonym_dict[normal_form]
    else:
        return []

def replace_word(sentence, target, replacement):
    target_normal = morph.parse(target)[0].normal_form
    
    tokens = tokenize_ukrainian(sentence)
    
    replaced = False
    new_tokens = []

    for i, token in enumerate(tokens):
        target_gender = None
        
        if re.match(r'\w+', token):
            parsed_word = morph.parse(token)[0]
            
            if parsed_word.normal_form == target_normal:
                case_and_number_grammemes = parsed_word.tag.grammemes
                target_pos = parsed_word.tag.POS

                if target_pos == 'NOUN':
                    target_gender = parsed_word.tag.gender

                replacement_parsed = morph.parse(replacement)
                matched_replacement = get_correct_parsed_result(replacement_parsed, target_pos, target_gender)
                # print(matched_replacement)
                # print(case_and_number_grammemes)

                if matched_replacement:
                    replacement_word = matched_replacement.inflect(case_and_number_grammemes)
                else:
                    # print('bad match')
                    return None
            
                if replacement_word:
                    replacement_word = replacement_word.word
                else:
                    case_and_number_grammemes = lower_grammar_restrictions(case_and_number_grammemes, target_gender)
                    replacement_word = matched_replacement.inflect(case_and_number_grammemes)
                    if replacement_word:
                        replacement_word = replacement_word.word
                    else:
                        # print('bad inflect')
                        return None
                

                if token.istitle():
                    replacement_word = replacement_word.capitalize()
                
                new_tokens.append(replacement_word)
                replaced = True
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    
    if not replaced:
        # print("Replacement failed for word:", target, replacement)
        return None
    # return ''.join(new_tokens)
    return new_tokens

def attack(text_ls, true_label, predictor, stop_words_set, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50):
    
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        return '', 0, orig_label, orig_label, 0
    else:
        num_queries = 1
        len_text = len(text_ls)

        pos_ls = criteria.get_pos(text_ls)
        
        # get importance score
        perturbable_indices = [i for i, tok in enumerate(text_ls) if is_word_token(tok) and tok not in stop_words_set]
        leave_1_texts = [text_ls[:i] + ['<oov>'] + text_ls[i+1:] for i in perturbable_indices]
        if len(leave_1_texts) == 0:
            print('no words for pertubation', ''.join(text_ls))
            return '', 0, orig_label, orig_label, 0
    
        leave_1_probs = predictor(leave_1_texts)

        num_queries += len(leave_1_texts)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))).data.cpu().numpy()
        words_perturb = []

        for idx, score in sorted(zip(perturbable_indices, import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold:
                    words_perturb.append((idx, text_ls[idx]))
            except:
                print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

        # find synonyms
        synonyms_all = []
        for (position, word) in words_perturb:
            # Use your custom dictionary-based function
            synonyms = get_all_synonyms(word)
            synonyms = [synonym for synonym in synonyms if len(synonym.split(' '))==1] #TODO explore to replace word to phrase
            synonyms = synonyms[:synonym_num]

            if synonyms:
                synonyms_all.append((position, synonyms))

        # start replacing and attacking
        text_prime = text_ls.copy()
        text_cache = text_prime.copy()

        num_changed = 0

        for idx, synonyms in synonyms_all:
            new_texts = [replace_word(''.join(text_prime), text_prime[idx], synonym) for synonym in synonyms]
            new_texts = [x for x in new_texts if x is not None]
            if len(new_texts) == 0:
                print(f"no synonyms for word {text_prime[idx]}")
                continue 

            new_probs = predictor(new_texts)
            num_queries += len(new_texts)
            
            semantic_sims = sim_predictor.semantic_sim([''.join(text_cache)] * len(new_texts), [''.join(text) for text in new_texts])[0]
            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)

            new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            new_probs_mask *= (semantic_sims >= sim_score_threshold)

            # prevent incompatible pos
            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                                if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                break
            else:
                new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                        (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                if new_label_prob_min < orig_prob:
                    text_prime[idx] = synonyms[new_label_prob_argmin]
                    num_changed += 1
            text_cache = text_prime[:]
        return ''.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


def main():
    dataset_path = "/home/mudryi/phd_projects/synonym_attack/cross_domain_uk_reviews/test_reviews.csv" # "Which dataset to attack."
    nclasses = 5 # "How many classes for classification."
    target_model = 'xlm-roberta-base' # "Target models for text classification: fasttext, charcnn, word level lstm "
    target_model_path = "/home/mudryi/phd_projects/xml-roberta-finetune-reviews/trained_models/tmdk/model_tmdk_7_1000" #"pre-trained target model path"

    word_embeddings_path = None # "path to the word embeddings for the target model"
    counter_fitting_embeddings_path = None # "path to the counter-fitting embeddings we used to find synonyms"
    counter_fitting_cos_sim_path = None # "pre-compute the cosine similarity scores based on the counter-fitting embeddings"
    
    SBERT_path = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1' # "Path to the USE encoder cache."

    output_dir = 'adv_results_reviews_xml_roberta' # "The output directory where the attack results will be written."

    ## Model hyperparameters
    sim_score_window = 25 # "Text length or token number to compute the semantic similarity score")
    import_score_threshold = -1 # "Required mininum importance score.")
    sim_score_threshold = 0.7 # "Required minimum semantic similarity score.")
    synonym_num = 100 # "Number of synonyms to extract"
    batch_size = 32 # "Batch size to get prediction"
    data_size = 2000 # "Data size to create adversaries" reviews have 9663 records
    perturb_ratio = 0 # "Whether use random perturbation for ablation study")
    max_seq_length = 256 # "max sequence length for BERT target model")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        print("Output directory ({}) already exists and is not empty.".format(output_dir))
    else:
        os.makedirs(output_dir, exist_ok=True)

    # get data to attack
    texts, labels = read_corpus(dataset_path)
    data = list(zip(texts, labels))

    data = data[:data_size] # choose how many samples for adversary
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    model = AutoModelForSequenceClassification.from_pretrained(target_model_path, num_labels=nclasses)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(target_model)

    def predictor(texts):
        """
        texts: a list of strings, e.g. ["This is a test", "Another sample"]
        Returns: a torch.Tensor of shape (batch_size, nclasses) with probabilities
        """
        if len(texts) > 0 and isinstance(texts[0], str):
            texts = [texts]

        # Now 'token_lists' is guaranteed to be a list of lists of tokens
        # Convert each token-list into one full string
        texts = [" ".join(tokens) for tokens in texts]

        # Tokenize with truncation/padding, move to GPU if available
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(device)
        
        with torch.no_grad():
            output = model(**inputs)
        
        # Output logits of shape [batch_size, nclasses]
        logits = output.logits
        
        # Convert logits -> probabilities
        probs = torch.softmax(logits, dim=1)
        return probs
    
    # predictor = model.text_pred
    print("Model built!")

    # build the semantic similarity module
    sbert = SBERT(SBERT_path)

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    log_file = open(os.path.join(output_dir, 'results_log'), 'a')

    stop_words_set = criteria.get_stopwords()
    print('Start attacking!')
    for idx, (text, true_label) in tqdm(enumerate(data), total=len(data), desc="Processing samples"):

        true_label = true_label - 1
        

        new_text, num_changed, orig_label, new_label, num_queries = attack(text, true_label, predictor, stop_words_set, sim_predictor=sbert,
                                                                           sim_score_threshold=sim_score_threshold,
                                                                           import_score_threshold=import_score_threshold,
                                                                           sim_score_window=sim_score_window,
                                                                           synonym_num=synonym_num)

        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)
        if true_label != new_label:
            adv_failures += 1

        changed_rate = 1.0 * num_changed / len(text)

        if true_label == orig_label and true_label != new_label:
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(text))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)

    message = 'For target model {}: original accuracy: {:.3f}%, adv accuracy: {:.3f}%, ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}\n'.format(target_model,
                                                                     (1-orig_failures/len(data))*100,
                                                                     (1-adv_failures/len(data))*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries))
    print(message)
    log_file.write(message)

    with open(os.path.join(output_dir, 'adversaries.txt'), 'w') as ofile:
        for orig_text, adv_text, true_label, new_label in zip(orig_texts, adv_texts, true_labels, new_labels):
            ofile.write('orig sent ({}):\t{}\nadv sent ({}):\t{}\n\n'.format(true_label, orig_text, new_label, adv_text))

if __name__ == "__main__":
    main()