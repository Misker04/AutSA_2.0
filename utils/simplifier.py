def simplify_sentence(sentence):
    rules = {
        "what's up": "how are you",
        "gonna": "going to",
        "wanna": "want to",
        "yeah": "yes",
    }
    for key, val in rules.items():
        sentence = sentence.replace(key, val)
    return sentence
adef simplify_sentence(sentence):
    rules = {
        "what's up": "how are you",
        "gonna": "going to",
        "wanna": "want to",
        "yeah": "yes",
        "lol": "that’s funny",
        "idk": "I do not know",
        "btw": "by the way",
        "u": "you",
        "r": "are",
    }
    for key, val in rules.items():
        sentence = sentence.lower().replace(key, val)
    return sentence.capitalize()
