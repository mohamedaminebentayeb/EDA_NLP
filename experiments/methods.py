from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional
import keras.layers as layers
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import math
import time
import numpy as np
import random
from random import randint
random.seed(3)
import datetime, re, operator
from random import shuffle
from time import gmtime, strftime
import gc

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #get rid of warnings
from os import listdir
from os.path import isfile, join, isdir
import pickle

#import data augmentation methods
from nlp_aug import *



import numpy as np
import random
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import gensim.downloader as api

# Download required NLTK resources if not already available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')  # Open Multilingual WordNet

# Load a small pre-trained Word2Vec model
# For production, consider downloading this in advance or using a smaller model
print("Loading pre-trained Word2Vec model (this might take a moment)...")
word2vec_model = None
try:
    # Using a smaller model (glove-twitter-25) that's faster to load
    word2vec_model = api.load("glove-twitter-25")
    print("Word2Vec model loaded successfully")
except Exception as e:
    print(f"Could not load Word2Vec model: {e}")
    print("Will fall back to WordNet only")

# Function to compute TF-IDF scores for a dataset
def compute_tfidf(sentences):
    """
    Compute TF-IDF scores for a list of sentences.
    
    Args:
        sentences (list): List of strings/sentences
        
    Returns:
        dict: Dictionary mapping words to their average TF-IDF scores
    """
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(lowercase=True, stop_words=None)
    
    # Fit and transform the sentences
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate average TF-IDF score for each word
    word_scores = {}
    for i, word in enumerate(feature_names):
        # Get the TF-IDF score for this word across all documents
        scores = tfidf_matrix[:, i].toarray().flatten()
        # Store the average score
        word_scores[word] = np.mean(scores)
    
    return word_scores

# Get WordNet synonyms with matching POS
def get_wordnet_synonyms(word, pos=None):
    """
    Get synonyms for a word that match its part of speech.
    
    Args:
        word (str): The word to find synonyms for
        pos (str): Part of speech tag (optional)
        
    Returns:
        list: List of synonyms
    """
    synonyms = []
    
    # Convert NLTK POS tag to WordNet POS tag
    wn_pos = None
    if pos:
        if pos.startswith('J'):
            wn_pos = wn.ADJ
        elif pos.startswith('V'):
            wn_pos = wn.VERB
        elif pos.startswith('N'):
            wn_pos = wn.NOUN
        elif pos.startswith('R'):
            wn_pos = wn.ADV
    
    # Get synsets based on POS if available
    if wn_pos:
        synsets = wn.synsets(word, pos=wn_pos)
    else:
        synsets = wn.synsets(word)
    
    # Collect all lemma names (synonyms)
    for synset in synsets:
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and synonym not in synonyms:
                synonyms.append(synonym)
    
    return synonyms

# Get similar words using Word2Vec
def get_word2vec_similar(word, topn=10):
    """
    Get similar words using Word2Vec model.
    
    Args:
        word (str): The word to find similar words for
        topn (int): Number of similar words to retrieve
        
    Returns:
        list: List of similar words
    """
    similar_words = []
    if word2vec_model and word in word2vec_model:
        try:
            # Get similar words and extract just the words (not the similarity scores)
            similar_words = [w for w, _ in word2vec_model.most_similar(word, topn=topn)]
        except:
            pass
    return similar_words

# Get the best replacement for a word considering both WordNet and Word2Vec
def get_best_replacement(word, pos=None):
    """
    Get the best replacement for a word using both WordNet and Word2Vec.
    
    Args:
        word (str): The word to replace
        pos (str): Part of speech tag
        
    Returns:
        str or None: Replacement word or None if no replacement found
    """
    # Try to get WordNet synonyms first (more context-aware)
    synonyms = get_wordnet_synonyms(word, pos)
    
    # If no synonyms found in WordNet, try Word2Vec
    if not synonyms and word2vec_model:
        synonyms = get_word2vec_similar(word)
    
    # Return a random synonym if any were found
    if synonyms:
        return random.choice(synonyms)
    return None

###################################################
######### loading folders and txt files ###########
###################################################

#loading a pickle file
def load_pickle(file):
	return pickle.load(open(file, 'rb'))

#create an output folder if it does not already exist
def confirm_output_folder(output_folder):
	if not os.path.exists(output_folder):
	    os.makedirs(output_folder)

#get full image paths
def get_txt_paths(folder):
    txt_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and '.txt' in f]
    if join(folder, '.DS_Store') in txt_paths:
        txt_paths.remove(join(folder, '.DS_Store'))
    txt_paths = sorted(txt_paths)
    return txt_paths

#get subfolders
def get_subfolder_paths(folder):
    subfolder_paths = [join(folder, f) for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)]
    if join(folder, '.DS_Store') in subfolder_paths:
        subfolder_paths.remove(join(folder, '.DS_Store'))
    subfolder_paths = sorted(subfolder_paths)
    return subfolder_paths

#get all image paths
def get_all_txt_paths(master_folder):

    all_paths = []
    subfolders = get_subfolder_paths(master_folder)
    if len(subfolders) > 1:
        for subfolder in subfolders:
            all_paths += get_txt_paths(subfolder)
    else:
        all_paths = get_txt_paths(master_folder)
    return all_paths

###################################################
################ data processing ##################
###################################################

import re
import json
import time
import asyncio
import random

# --- Partie 1 : Fonctions LLM et utilitaires ---

def extract_json_from_text(response_text):
    """
    Recherche et extrait un objet JSON valide à partir du texte de la réponse.
    Cherche la partie entourée par des backticks, puis tente de charger le JSON.
    Retourne l'objet JSON si trouvé, sinon None.
    """
    # Rechercher le JSON entouré de backticks
    json_pattern = r'```json\n\{.*\}\n```'
    match = re.search(json_pattern, response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
        # Enlever les backticks et les éventuelles nouvelles lignes autour
        json_str = json_str.strip('```json\n').strip('\n```')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Erreur lors du parsing du JSON extrait.")
            return None
    else:
        print("Aucun JSON valide trouvé dans la réponse.")
        return None

def get_only_chars(line):
    """
    Nettoie la ligne en retirant les caractères non alphabétiques et en normalisant la casse.
    """
    clean_line = ""
    line = line.replace("’", "").replace("'", "").replace("-", " ").replace("\t", " ").replace("\n", " ").lower()
    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '
    clean_line = re.sub(' +', ' ', clean_line).strip()
    return clean_line

async def simulate_call_mistral(prompt):
    """
    Simule un appel asynchrone vers une API LLM.
    Cette fonction fait un sleep de 1.5 secondes et retourne une réponse factice.
    """
    await asyncio.sleep(1.5)
    # Réponse factice : on retourne le prompt modifié pour simuler une transformation
    fake_response = f"```json\n{{\"sentences\": \"{prompt.strip()} - texte modifié\"}}\n```"
    return fake_response

async def call_mistral(prompt, sentence):
    """
    Appelle la fonction simulate_call_mistral pour simuler une réponse de l'API LLM.
    Extrait ensuite le JSON de la réponse et retourne la valeur associée à 'sentences'.
    """
    try:
        response_text = await simulate_call_mistral(prompt)
        response_json = extract_json_from_text(response_text)
        if response_json and 'sentences' in response_json:
            return response_json['sentences']
        else:
            return None
    except Exception as e:
        print(f"Erreur lors de l'appel à Mistral : {e}")
        return None

async def synonym_replacement(sentence, n):
    """
    Génère une augmentation de données en remplaçant exactement n mots de la phrase par des synonymes via LLM.
    """
    prompt = f"""
Effectue une augmentation de données en remplaçant exactement {n} mots de la phrase suivante par des synonymes.
Les synonymes doivent préserver le sens original de la phrase.
Retourne les résultats sous forme de JSON. Exemple :
{{"sentences": "phrase modifiée"}}

Voici la phrase à modifier : {sentence}
    """
    return await call_mistral(prompt, sentence)

async def random_deletion(sentence, p):
    """
    Génère une augmentation de données en supprimant certains mots de la phrase avec une probabilité de p via LLM.
    """
    prompt = f"""
Effectue une augmentation de données en supprimant certains mots de la phrase suivante avec une probabilité de {p}.
Les mots supprimés doivent être choisis de façon à préserver le sens global de la phrase.
Retourne les résultats sous forme de JSON. Exemple :
{{"sentences": "phrase modifiée"}}

Voici la phrase à modifier : {sentence}
    """
    return await call_mistral(prompt, sentence)

async def random_swap(sentence, n):
    """
    Génère une augmentation de données en échangeant exactement n mots dans la phrase via LLM.
    """
    prompt = f"""
Effectue une augmentation de données en échangeant exactement {n} mots dans la phrase suivante tout en préservant le sens global.
Retourne les résultats sous forme de JSON. Exemple :
{{"sentences": "phrase modifiée"}}

Voici la phrase à modifier : {sentence}
    """
    return await call_mistral(prompt, sentence)

async def random_insertion(sentence, n):
    """
    Génère une augmentation de données en insérant exactement n mots dans la phrase via LLM.
    """
    prompt = f"""
Effectue une augmentation de données en insérant exactement {n} mots dans la phrase suivante tout en préservant son sens.
Assure-toi que les mots insérés s’intègrent naturellement dans la phrase.
Retourne les résultats sous forme de JSON. Exemple :
{{"sentences": "phrase modifiée"}}

Voici la phrase à modifier : {sentence}
    """
    return await call_mistral(prompt, sentence)

# --- Partie 2 : Adaptation du deuxième code pour utiliser le LLM ---

# 1. Synonym Replacement (SR)
async def gen_sr_aug(train_orig, output_file, alpha_sr, n_aug):
    """
    Augmente les données en remplaçant par synonymes via appel LLM.
    Les noms de fonction et paramètres restent inchangés.
    """
    writer = open(output_file, 'w', encoding='utf-8')
    lines = open(train_orig, 'r', encoding='utf-8').readlines()
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        label, sentence = parts[0], parts[1]
        aug_sentences = set()
        
        for _ in range(n_aug):
            n = max(1, int(alpha_sr * len(sentence.split())))
            result = await synonym_replacement(sentence, n)
            if result:
                aug_sentences.add(result)
        
        for aug_sentence in aug_sentences:
            writer.write(f"{label}\t{aug_sentence}\n")
    
    writer.close()
    print(f"Finished SR for {train_orig} to {output_file} with alpha {alpha_sr}")

# 2. Random Insertion (RI)
async def gen_ri_aug(train_orig, output_file, alpha_ri, n_aug):
    """
    Augmente les données en insérant des mots via appel LLM.
    Les noms de fonction et paramètres restent inchangés.
    """
    writer = open(output_file, 'w', encoding='utf-8')
    lines = open(train_orig, 'r', encoding='utf-8').readlines()
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        label, sentence = parts[0], parts[1]
        aug_sentences = set()
        
        for _ in range(n_aug):
            n = max(1, int(alpha_ri * len(sentence.split())))
            result = await random_insertion(sentence, n)
            if result:
                aug_sentences.add(result)
        
        for aug_sentence in aug_sentences:
            writer.write(f"{label}\t{aug_sentence}\n")
    
    writer.close()
    print(f"Finished RI for {train_orig} to {output_file} with alpha {alpha_ri}")

# 3. Random Swap (RS)
async def gen_rs_aug(train_orig, output_file, alpha_rs, n_aug):
    """
    Augmente les données en échangeant des mots via appel LLM.
    Les noms de fonction et paramètres restent inchangés.
    """
    writer = open(output_file, 'w', encoding='utf-8')
    lines = open(train_orig, 'r', encoding='utf-8').readlines()
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        label, sentence = parts[0], parts[1]
        aug_sentences = set()
        
        for _ in range(n_aug):
            n = max(1, int(alpha_rs * len(sentence.split())))
            result = await random_swap(sentence, n)
            if result:
                aug_sentences.add(result)
        
        for aug_sentence in aug_sentences:
            writer.write(f"{label}\t{aug_sentence}\n")
    
    writer.close()
    print(f"Finished RS for {train_orig} to {output_file} with alpha {alpha_rs}")

# 4. Random Deletion (RD)
async def gen_rd_aug(train_orig, output_file, alpha_rd, n_aug):
    """
    Augmente les données en supprimant certains mots via appel LLM.
    Les noms de fonction et paramètres restent inchangés.
    """
    writer = open(output_file, 'w', encoding='utf-8')
    lines = open(train_orig, 'r', encoding='utf-8').readlines()
    
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        label, sentence = parts[0], parts[1]
        aug_sentences = set()
        
        for _ in range(n_aug):
            result = await random_deletion(sentence, alpha_rd)
            if result:
                aug_sentences.add(result)
        
        for aug_sentence in aug_sentences:
            writer.write(f"{label}\t{aug_sentence}\n")
    
    writer.close()
    print(f"Finished RD for {train_orig} to {output_file} with alpha {alpha_rd}")


###################################################
##################### model #######################
###################################################

#building the model in keras
def build_model(sentence_length, word2vec_len, num_classes):
	model = None
	model = Sequential()
	model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(sentence_length, word2vec_len)))
	model.add(Dropout(0.5))
	model.add(Bidirectional(LSTM(32, return_sequences=False)))
	model.add(Dropout(0.5))
	model.add(Dense(20, activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#print(model.summary())
	return model

#building the cnn in keras
def build_cnn(sentence_length, word2vec_len, num_classes):
	model = None
	model = Sequential()
	model.add(layers.Conv1D(128, 5, activation='relu', input_shape=(sentence_length, word2vec_len)))
	model.add(layers.GlobalMaxPooling1D())
	model.add(Dense(20, activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#one hot to categorical
def one_hot_to_categorical(y):
    assert len(y.shape) == 2
    return np.argmax(y, axis=1)

def get_now_str():
    return str(strftime("%Y-%m-%d_%H:%M:%S", gmtime()))

