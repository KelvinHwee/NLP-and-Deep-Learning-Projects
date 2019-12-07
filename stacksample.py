#==============================================================================
#=====     change current working directory and some configurations
#==============================================================================
import os
path = r'C:\Users\eight\Desktop\Kelvin HDD\5. Kaggle\Kaggle competition\11. StackSample - 10% of Stack Overflow Q&A'
os.chdir(path)
os.getcwd()

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 50)

#==============================================================================
#=====     read datasets and looking at the contents
#==============================================================================
data_qns = pd.read_csv('Questions.csv', encoding = 'latin-1')
data_ans = pd.read_csv('Answers.csv', encoding = 'latin-1')
data_tags = pd.read_csv('Tags.csv', encoding = 'latin-1')

data_qns.head(); data_qns.shape
data_ans.head(); data_ans.shape  #'ParentId' links back to the 'Id' of the questions dataset
data_tags.head(); data_tags.shape

# order the datasets by 'Id' for the questions dataset, 'ParentId' for answers datasets
data_ans.sort_values(by = 'ParentId').head(200)


#==============================================================================
#=====     perform cleaning of the text for answers and questions
#==============================================================================
# look at a data sample
list(data_qns)
text = data_qns.Body[0]

# create a function to do the cleaning
def clear_tags_clean_up(text):
	import re		
	text = re.sub("<pre>.*</pre>", "", text, flags = re.DOTALL) # removes the codes
	text = re.sub("<.*?>", "", text) # removes unnecessary tags
	#text = re.sub("PageContent.*PageContent", "", text)
	text = re.sub("\r", "", text)
	text = re.sub("[\s]{2,}", " ", text)
	return re.sub("\n", "", text) # removes newline characters
	
# clean up the text body for both datasets
data_qns['cleaned_qns'] = data_qns.Body.apply(clear_tags_clean_up)
data_ans['cleaned_ans'] = data_ans.Body.apply(clear_tags_clean_up)


#==============================================================================
#=====     feature engineering
#==============================================================================
# lets include the feature about the len of the text
data_qns['length'] = data_qns.cleaned_qns.apply(len)
data_ans['length'] = data_ans.cleaned_ans.apply(len)

data_qns.head(50)
data_ans.head(50)


#==============================================================================
#=====     create some visualisations
#==============================================================================
# for the "Tags" dataset, we take a look at visualisation for the top tags
import matplotlib.pyplot as plt
from collections import Counter
count = Counter([i for i in data_tags.Tag]).most_common(50)

x = []; y = []; z = []
for i in range(len(count)):
	x.append(count[i][0])
	y.append(count[i][1])
	z.append((count[i][1])/100)
	
plt.figure(figsize = (14,8))
plt.scatter(x,y, c = y, s = z, alpha = 0.75)
plt.xticks(rotation = 60)

# lets take a look at the frequency of questions and answers against their scores
import seaborn as sns
answers = data_ans.sort_values(by = "Score", ascending = False).head(50)
questions = data_qns.sort_values(by = "Score", ascending = False).head(50)

fig, ax = plt.subplots(2,1, figsize = (14,8), sharex = True)
sns.distplot(answers.Score, bins = 50, ax = ax[0], color = 'navy')
sns.distplot(questions.Score, bins = 50, ax = ax[1], color = 'red')

# we take a look at the score against the len of the text, for the answers dataset
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (12,10))
sns.lineplot(data_ans.Score, data_ans.length)

plt.figure(figsize = (12,10))
sns.lineplot(data_ans.length, data_ans.Score)


#==============================================================================
#=====     start using SpaCy + text pre-processing
#==============================================================================
# load packages and “SpaCy”
import re
import spacy
nlp = spacy.load('en_core_web_sm')  # nlp = spacy.load('en_core_web_lg')

# due to memory constraints, we limit the number of documents to read in
# create a ranked column based on the longest text length, but based on scores
data_qns_ranked = data_qns[(data_qns['Score'] > 200)  & 
						   (data_qns['length'] < 5000) &
						   (data_qns['length'] > 2000)]

data_qns_ranked = data_qns_ranked.sort_values(by = ['Score', 'length'], ascending = False)
data_qns_ranked_truncated = data_qns_ranked.head(1)  # LIMIT NUMBER OF TEXT
data_qns_ranked_truncated

# for the answers dataset
data_ans_ranked = data_ans.sort_values(by =['length'], ascending = False)
data_ans_ranked_truncated = data_ans_ranked.head(1)  # LIMIT NUMBER OF TEXT

# lets collapse all the text into one variable
all_sentences = [text for text in data_qns_ranked_truncated.cleaned_qns]  
doc = nlp(str(all_sentences))

# performing sentence tokenization using "SpaCy"
sentence_list=[]
for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
	sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))  # remove punctuations and special characters

len(sentence_list) # XXXX sentences

# create the corpus for stopwords
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words('english'))

# remove the stopwords in each sentence
from nltk.tokenize import word_tokenize

cleaned_sentences = []
for i in range(len(sentence_list)):
	cleaned_sentences.append(' '.join(word for word in word_tokenize(sentence_list[i]) if word not in eng_stopwords and len(word) > 2))		
	
cleaned_sentences

#==============================================================================
#=====     start creating "GloVe" word embeddings
#==============================================================================
# we use "GloVe" word embeddings which are vector representation of words
# word embeddings will be used to create vectors for our sentences
# we could have used bag-of-words or TF-IDF approaches to create features for our sentences
# but these methods ignore the order of the words (and the number of features is usually large)

# we load a pre-trained Wikipedia 2014 + Gigaword 5 GloVe vectors
import numpy as np

word_embeddings = {}  # create a dictionary
f = open('glove.6B.100d.txt', encoding = 'utf-8')
for line in f:
	values = line.split() 
	word = values[0]  # this is the "key"
	coefs = np.asarray(values[1:], dtype='float32')  # collect the values into the dictionary
	word_embeddings[word] = coefs # create the dictionary
f.close()

len(word_embeddings) # word vectors for 400,000 different terms stored in the dictionary


#==============================================================================
#=====     create vectors for our sentences
#==============================================================================
# fetch vectors (each of size 100 elements) for the constituent words in a sentence
import numpy as np

sentence_vectors = []
for i in cleaned_sentences:
	if len(i) != 0:
		# use the "get" method to get the "value", given the "key" 
		# (key is the word from the sentence)
		# furthermore, these matrices (or coefficients) are summed
		v = sum([word_embeddings.get(w, np.zeros(100,)) for w in i.split()]) / (len(i.split()) + 0.001)
	else:
		v = np.zeros(100,)
	sentence_vectors.append(v)

# each item is the collective importance of the sentence, 
# based on the importance of the words it contains
sentence_vectors  


#==============================================================================
#=====     create the similarity matrix, to find similarity between sentences
#==============================================================================
# we use the cosine similarity approach for this challenge
from sklearn.metrics.pairwise import cosine_similarity

# create similarity matrix
similarity_matrix = np.zeros([len(sentence_list), len(sentence_list)])

# initialize the similarity matrix with cosine similarity scores
for i in range(len(sentence_list)-1):
	for j in range(len(sentence_list)-1):
		if i != j:
			similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100),
					                                    sentence_vectors[j].reshape(1,100))[0][0]

# cosine_similarity(sentence_vectors[2].reshape(1,100), sentence_vectors[4].reshape(1,100))[0,0]


#==============================================================================
#=====     apply the PageRank Algorithm
#==============================================================================
# we attempt to convert the similarity matrix into a graph
# nodes of the graph will represent the sentences 
# edges will represent the similarity scores between the sentences
# on this graph, we will apply the PageRank algorithm to arrive at the sentence rankings
import networkx as nx

nx_graph = nx.from_numpy_array(similarity_matrix)
scores = nx.pagerank(nx_graph)

# extract the top 5 sentences as the summary
num_to_extract = 5
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentence_list)), reverse=True)

ranked_output = []
for i in range(num_to_extract):
	ranked_output.append(ranked_sentences[i][1])
	
import pandas as pd
pd.set_option('display.max_colwidth', -1)
data_qns_ranked_truncated
ranked_output
	
	
	
	
	
	
	



