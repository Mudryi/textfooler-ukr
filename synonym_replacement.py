import re


INTERCHANGEABLE_POS = {
    'PRCL': {'ADVB'},
    'ADVB': {'PRCL'}}

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


def get_correct_parsed_result(parsing_results, target_pos, target_gender=None):
    acceptable_pos = {target_pos} | INTERCHANGEABLE_POS.get(target_pos, set())

    for result in parsing_results:
        if result.tag.POS in acceptable_pos and target_gender and result.tag.gender == target_gender:
            return result

    for result in parsing_results:
        if result.tag.POS in acceptable_pos:
            return result

    return None


def lower_grammar_restrictions(grammemes):
    grammemes_to_remove = ['Refl', 'compb', 'COMP', 'Qual']
    
    new_grammemes = set(item for item in list(grammemes) if item not in grammemes_to_remove)
    return new_grammemes


def stepwise_inflect(parse, target_grammemes, 
                     preferred_order=('plur', 'sing', 'femn', 'masc', 'neut', 'nomn', 'accs', 'gent', 'datv', 'loct', 'ablt', 'anim', 'inan')):
    current = parse
    applied = set()

    sorted_grammemes = sorted(target_grammemes, key=lambda g: preferred_order.index(g) if g in preferred_order else len(preferred_order))
    
    for gram in sorted_grammemes:
        attempt = current.inflect(applied | {gram})
        if attempt is not None:
            current = attempt
            applied |= {gram}
    return current


def replace_word(sentence, target, replacement, morph):
    target_normal = morph.parse(target)[0].normal_form
    
    tokens = tokenize_ukrainian(sentence)
    
    replaced = False
    new_tokens = []

    for i, token in enumerate(tokens):        
        if re.match(r'\w+', token):
            parsed_word = morph.parse(token)[0]
            
            if parsed_word.normal_form == target_normal:
                target_pos = parsed_word.tag.POS

                target_gender = parsed_word.tag.gender if target_pos in ('NOUN', 'ADJF') else None

                replacement_parsed = morph.parse(replacement)

                matched_replacement = get_correct_parsed_result(replacement_parsed, target_pos, target_gender)

                if not matched_replacement:
                    print('bad match', target, replacement)
                    # print(target_pos, target_gender)
                    # print(replacement_parsed)
                    return None
                
                grammemes = parsed_word.tag.grammemes 
                cleaned_grammemes = lower_grammar_restrictions(grammemes)
                
                replacement_inflected = stepwise_inflect(matched_replacement, cleaned_grammemes)
                
                if not replacement_inflected:
                    print(f'bad inflect for replacement: {target} -> {replacement}')
                    return None
                
                replacement_word = replacement_inflected.word
                
                if token.istitle():
                    replacement_word = replacement_word.capitalize()
                
                new_tokens.append(replacement_word)
                replaced = True
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)
    
    if not replaced:
        print(f'no replacement for {target}')
        return None
    return new_tokens