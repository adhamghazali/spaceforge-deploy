
def detect_bad_words(prompt,list_of_bad_words):
    words=prompt.split(' ')
    for word in words:
        if word in File:
            return True
    return False