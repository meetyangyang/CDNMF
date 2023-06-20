#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# import nltk
# nltk.download('wordnet')

# 获取单词的词性
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatizing(data_samples_raw):
    lemmatizer = WordNetLemmatizer()
    data_samples_cleaned=[]
    for ind_doc in range(0,len(data_samples_raw)):

        sentence=data_samples_raw[ind_doc].lower()
        tokens = word_tokenize(sentence)  # 分词
        tagged_sent = pos_tag(tokens)     # 获取单词词性
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(lemmatizer.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
    #     data_samples_cleaned.append(lemmas_sent)
        data_samples_cleaned.append(' '.join(lemmas_sent))
    return(data_samples_cleaned)

