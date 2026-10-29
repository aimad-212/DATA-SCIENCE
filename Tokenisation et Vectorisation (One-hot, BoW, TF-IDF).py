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

# =====================================================
# 8. TF-IDF
# =====================================================

def tfidf_doc(doc_tokens, corpus, vocab):
    """
    Calcule le vecteur TF-IDF d'un document.

    TF-IDF(term, doc) = TF(term, doc) * IDF(term)
    """
    vector = []

    for word in vocab:
        term_frequency = tf(word, doc_tokens)
        inverse_document_frequency = idf(word, corpus)

        weight = term_frequency * inverse_document_frequency

        vector.append(weight)

    return np.array(vector, dtype=float)

# =====================================================
# 9. APPLICATION AUX QUATRE DOCUMENTS
# =====================================================

one_hot_matrix = np.array([
    one_hot_doc(tokens, vocab)
    for tokens in tokenized_corpus
])

bow_matrix = np.array([
    bow_doc(tokens, vocab)
    for tokens in tokenized_corpus
])

tfidf_matrix = np.array([
    tfidf_doc(tokens, tokenized_corpus, vocab)
    for tokens in tokenized_corpus
])


# =====================================================
# 10. AFFICHAGE DES TABLEAUX
# =====================================================

document_names = ["Doc1", "Doc2", "Doc3", "Doc4"]


def display_matrix(title, matrix):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"{'Mot':<18}", end="")

    for name in document_names:
        print(f"{name:>12}", end="")

    print()

    for j, word in enumerate(vocab):
        print(f"{word:<18}", end="")

        for i in range(len(corpus)):
            value = matrix[i, j]

            if title == "TF-IDF":
                print(f"{value:>12.4f}", end="")
            else:
                print(f"{int(value):>12}", end="")

        print()


display_matrix("ONE-HOT ENCODING", one_hot_matrix)
display_matrix("BAG OF WORDS", bow_matrix)
display_matrix("TF-IDF", tfidf_matrix)

# =====================================================
# 11. TEST SUR DOC1
# =====================================================

tokens_doc1 = tokenized_corpus[0]

one_hot_doc1 = one_hot_doc(tokens_doc1, vocab)
bow_doc1 = bow_doc(tokens_doc1, vocab)
tfidf_doc1 = tfidf_doc(
    tokens_doc1,
    tokenized_corpus,
    vocab
)

print("\n========== RESULTATS DOC1 ==========")

print("Tokens :", len(tokens_doc1))
print("Vocabulaire :", len(vocab))

print("\nOne-hot Doc1 :")
print(one_hot_doc1)

print("\nBoW Doc1 :")
print(bow_doc1)

print("\nTF-IDF Doc1 :")
print(np.round(tfidf_doc1, 4))


# =====================================================
# 12. VERIFICATIONS
# =====================================================

assert len(vocab) == 65

assert np.sum(bow_doc1) == len(tokens_doc1)

assert np.sum(one_hot_doc1) == len(set(tokens_doc1))

assert np.all(one_hot_matrix >= 0)
assert np.all(bow_matrix >= 0)
assert np.all(tfidf_matrix >= 0)

print("\nToutes les vérifications sont réussies.")