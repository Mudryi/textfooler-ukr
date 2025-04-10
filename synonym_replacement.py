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


def get_correct_parsed_result(parsing_results, target_pos, target_gender=None):

    for result in parsing_results:
        if result.tag.POS == target_pos and target_gender and result.tag.gender == target_gender:
            return result

    for result in parsing_results:
        if result.tag.POS == target_pos:
            return result

    return None


def lower_grammar_restrictions(grammemes, target_gender):
    grammemes_to_remove = [str(target_gender), 'Refl', 'compb', 'COMP']
    
    new_grammemes = set(item for item in list(grammemes) if item not in grammemes_to_remove)
    return new_grammemes


def replace_word(sentence, target, replacement, morph):
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

                if target_pos == 'NOUN' or target_pos == 'ADJF':
                    target_gender = parsed_word.tag.gender

                replacement_parsed = morph.parse(replacement)
                matched_replacement = get_correct_parsed_result(replacement_parsed, target_pos, target_gender)

                if matched_replacement:
                    replacement_word = matched_replacement.inflect(case_and_number_grammemes)
                else:
                    print('bad match')
                    return None
            
                if replacement_word:
                    replacement_word = replacement_word.word
                else:
                    case_and_number_grammemes = lower_grammar_restrictions(case_and_number_grammemes, target_gender)
                    replacement_word = matched_replacement.inflect(case_and_number_grammemes)
                    if replacement_word:
                        replacement_word = replacement_word.word
                    else:
                        print('bad inflect')
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
        return None
    return new_tokens