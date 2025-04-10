import json 

deletions = {
    "чоловік": ["жінка", "подружжя", "дружина", "подруга"],
    "жінка": ["чоловік"],
    "місце": ["ід"],
    "стояти": ["клячати", "писатися"],
    "грати": ["зображати"],
    "люди": ["мир"],
    "смугастий": ["із смугами"],
    "сонце": ["приязнь"],
    "вода": ["курорт"],
    "хлопець": ["дівчина"],
    "дівчина": ["коханий"],
    "маленький": ["великий"],
    "білий": ["чорний"],
    "рука": ["у десна"],
    "стіл": ["харч"],
    "рожевий": ["радісний"],
    "чорний": ["білий"],
    "сидіти": ["проживати", "(на", "мешкати"],
    "молодий": ["старий"],
    "собака": ["змія"],
    "купити": ["ошукати", "обманути"],
    "добрий": ["жалісний"],
    "проблема": ["інтерес"],
    "великий": ["страшний"],
    "зручний": ["зграбний", "звинний"],
    "рекомендувати": ['познайомити', 'знайомити', 'представити', 'представляти', 'познакомити', 'зазнайомлювати', 'знакомити', 'зазнайомити']}

remove_list = ["пестл", "розм", "від", "за", "жм", "як зв", "жарт", "тйж-ба"]

def read_and_process_synonym_dict(path_to_dict):
    with open(path_to_dict, 'r', encoding='utf-8') as file:
        dict_ = json.load(file)
    
    dict_processed = {}
    for i in dict_:
        if len(i["synsets"]) == 0:
            continue
        
        if i["lemma"].lower() in dict_processed:
            for synset in i["synsets"]:
                dict_processed[i["lemma"].lower()].extend(synset["clean"])
        else:
            dict_processed[i["lemma"].lower()] = []
            
            for synset in i["synsets"]:
                dict_processed[i["lemma"].lower()].extend(synset["clean"])
        
        dict_processed[i["lemma"].lower()] = list(set(dict_processed[i["lemma"].lower()]))
    return dict_processed

def remove_synonyms(synonym_dict, deletions):
    for key, values_to_remove in deletions.items():
        if key in synonym_dict:  # Check if the key exists in synonym_dict
            for value in values_to_remove:
                if value in synonym_dict[key]:  # Check if the value exists in the list
                    synonym_dict[key].remove(value)

def read_and_clean_synonym_dict(path_to_dict):
    synonym_dict = read_and_process_synonym_dict(path_to_dict)
    remove_synonyms(synonym_dict, deletions)

    for key in synonym_dict:
        synonym_dict[key] = [word for word in synonym_dict[key] if word not in remove_list]
        return synonym_dict