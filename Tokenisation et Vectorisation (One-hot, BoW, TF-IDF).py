import re
import numpy as np
from collections import Counter

# =====================================================
# 1. CORPUS
# =====================================================

doc1 = """
London is the capital of the United Kingdom.
It has a rich history and many landmarks.
Tourists love visiting Big Ben and the Tower of London.
"""

doc2 = """
New York is a bustling city in the United States.
Known for its skyscrapers and diverse culture.
Central Park is a green oasis in the urban jungle.
"""

doc3 = """
Paris is the city of lights in France.
Famous for the Eiffel Tower and fine cuisine.
The Louvre Museum houses world-renowned art.
"""

doc4 = """
Tokyo is the vibrant capital of Japan.
It blends tradition with modern technology.
Shibuya Crossing is one of the busiest intersections in the world.
"""

corpus = [doc1, doc2, doc3, doc4]

# =====================================================
# 2. TOKENISATION
# =====================================================

def tokenize_doc(doc):
    """
    Convertit le texte en minuscules,
    supprime la ponctuation et extrait les mots.
    Les mots composés avec un tiret sont séparés.
    """
    return re.findall(r"[a-z]+", doc.lower())


# Tokenisation des quatre documents
tokenized_corpus = [tokenize_doc(doc) for doc in corpus]

for i, tokens in enumerate(tokenized_corpus, start=1):
    print(f"Doc{i} :", tokens)
    print("Nombre de tokens :", len(tokens))
    print()

# =====================================================
# 3. CONSTRUCTION DU VOCABULAIRE
# =====================================================

# On conserve l'ordre de première apparition.
vocab = list(dict.fromkeys(
    token
    for tokens in tokenized_corpus
    for token in tokens
))

print("Vocabulaire :", vocab)
print("Taille du vocabulaire :", len(vocab))

# =====================================================
# 4. ONE-HOT ENCODING
# =====================================================

def one_hot_doc(tokens, vocab):
    """
    Retourne un vecteur binaire :
    1 si le mot est présent, 0 sinon.
    """
    token_set = set(tokens)

    return np.array(
        [1 if word in token_set else 0 for word in vocab],
        dtype=int
    )


# =====================================================
# 6. TERM FREQUENCY (TF)
# =====================================================

def tf(term, tokens):
    """
    TF = nombre d'occurrences du terme dans le document.
    """
    return tokens.count(term)


# =====================================================
# 7. INVERSE DOCUMENT FREQUENCY (IDF)
# =====================================================

def idf(term, corpus):
    """
    IDF = ln(N / DF)
    N  : nombre total de documents
    DF : nombre de documents contenant le terme
    """
    N = len(corpus)

    df = sum(
        1 for tokens in corpus
        if term in set(tokens)
    )

    if df == 0:
        return 0.0

    return np.log(N / df)