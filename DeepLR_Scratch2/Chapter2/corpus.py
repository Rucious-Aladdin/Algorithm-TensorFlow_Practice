import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace(".", " .")
    words = text.split(" ")
    
    w_to_id = {}
    id_to_w = {}
    
    for word in words:
        if word not in w_to_id:
            new_id = len(w_to_id)
            w_to_id[word] = new_id
            id_to_w[new_id] = word
            
    corpus = np.array([w_to_id[w] for w in words])
    
    return corpus, w_to_id, id_to_w

if __name__ == "__main__":
    text = "Hello Guys."

    corpus, word_to_id, id_to_word = preprocess(text)
    print(corpus, word_to_id, id_to_word)

    text = "You say goodbye and I say hello."

    text = text.lower()
    text = text.replace(".", " .")
    print(text)
    words = text.split(" ")
    print(words)

    word_to_id = {}
    id_to_word = {}
        
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    print(id_to_word)
    print(word_to_id)
    corpus = [word_to_id[w] for w in words]
    corpus = np.array(corpus)
    print(corpus)