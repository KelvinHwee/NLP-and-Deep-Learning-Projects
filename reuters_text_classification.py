#==============================================================================
#===   change to the correct directory
#==============================================================================
import os
path = r'C:\Users\eight\Desktop\Kelvin HDD\3. Coursera\A. Projects\Reuters text classification'
os.chdir(path)
os.getcwd()

#==============================================================================
#===   unzip the tar.gz files
#==============================================================================
import tarfile
tar = tarfile.open('reuters21578.tar.gz')
tar.getnames()  # displays the names of the files within the tar file
tar.extractall()  # this will unzip the files into the current working directory

#==============================================================================
#===   some basic configuration settings
#==============================================================================
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#==============================================================================
#===   use beautiful soup 
#==============================================================================
import numpy as np
from bs4 import BeautifulSoup

num = len([name for name in os.listdir() if name.endswith('sgm')])
data = open('reut2-00{}.sgm'.format(0), 'r').read()
for i in np.arange(1,num):
	if i < 10:
		add_data = open('reut2-00{}.sgm'.format(i), 'r').read()
		data = str(data) + str(add_data)
	else:
		add_data = open('reut2-0{}.sgm'.format(i), 'r').read()
		data = str(data) + str(add_data)
		
	soup = BeautifulSoup(str(data))

#data = open('reut2-00{}.sgm'.format(0), 'r').read()
#soup = BeautifulSoup(str(data))

#=== using "find_all" 
# we can print out the titles of the articles
soup.find_all(["title"])  
len(soup.find_all(["title"]))  # there are 20841 different articles in total

# we can print out the titles of the articles
soup.find_all(["text"])[:5]

menu = []
for meat in soup.find_all(["text"]):
	menu.append(meat.contents)

len(menu) 	 	# there are 21578 text blocks (as implied in the title)
type(menu) 	 	# "menu" is a list object
len(str(menu)) 	# there are 19305475 characters in the entire menu

#==============================================================================
#===   we clean up the text in the "menu"
#==============================================================================
#===   remove all the tags, and replace "\\n" with "\n"
import re

new_menu = []  # this is a list of article items 
for i, item in enumerate(menu):
	text = re.sub(r"\\n", "\n", str(menu[i])) # replace "\\n" with "\n"
	text = re.sub("[<]+[/a-zA-Z]*[>]+", "", text) # replace the tags with space
	new_menu.append(text)

# we want to clean up the document set, "new_menu", which we first combine the 
# objects in the new_menu first
new_menu_str = new_menu[0]
for i in range(1,len(new_menu)-1):
	new_menu_str += new_menu[i]


#===   clean up of the string; remove punctuations, square brackets, space characters
#new_menu_str_cleaned = re.sub("[\n\d,.&'-<>;]\"+", " ", new_menu_str)  # removes punctuations
#new_menu_str_cleaned = re.sub("\]\[", " ", new_menu_str_cleaned)   # removes square brackets
#new_menu_str_cleaned = re.sub("[\s]{2,}", " ", new_menu_str_cleaned)  # removes excessive space characters

new_menu_str_cleaned = new_menu_str

# to check the words before replacing them, i.e. whether they are standalone words
re.findall("[\w]+(?:dlrs)[\w]+", new_menu_str_cleaned)
re.findall("[\w]+(?:cts)[\w]+", new_menu_str_cleaned)
re.findall("[\w]+(?:mln)[\w]+", new_menu_str_cleaned)
re.findall("[\w]+(?:shrs)[\w]+", new_menu_str_cleaned)
re.findall("[\w]+(?:shr)[\w]+", new_menu_str_cleaned)

#===   replace short hands to the full de-abbreviated forms
new_menu_str_cleaned = re.sub(r"\bdlrs\b", "dollars", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(r"\bcts\b", "cents", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(r"\bmln\b", "million", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(r"\bavgsmlns\b", "average millions", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(r"\b50mlndlr\b", "50 million dollars", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(r"\bshr\b", "shares", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(r"\blbs\b", "pounds", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(r"\breuter", "", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub("[,.]", "", new_menu_str_cleaned)  # removes separator for currency; not sentences

#==============================================================================
#===   part of speech tagging
#==============================================================================
#===   we do these for the sentences, as part of word_tokenization as well
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

sentences = sent_tokenize(new_menu_str_cleaned)
tagged_tokens = []
for s in sentences:
	tagged_tokens.append(nltk.pos_tag(word_tokenize(s)))

tagged_tokens
len(tagged_tokens)


#===   clean up the tags
# create a function to convert the tags, and to create "None" types if there is no match
def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}   # missing are: "DT", "IN", "."
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None

cleaned_tokens = []
for i in range(len(tagged_tokens)-1):
	for j in range(len(tagged_tokens[i])-1):
		if len(tagged_tokens[i][j][0]) > 2 and convert_tag(tagged_tokens[i][j][1]) != None:
			cleaned_tokens.append(tagged_tokens[i][j])

cleaned_tokens

# =============================================================================
# #===   clean up strings of punctuations and (excessive) space characters
# new_menu_str_cleaned = re.sub("[^a-zA-Z0-9 ]", " ", new_menu_str_cleaned)  # removes punctuations
# new_menu_str_cleaned = re.sub("[\s]{2,}", " ", new_menu_str_cleaned)  # removes excessive space characters
# =============================================================================

#===   clean up the dates in the string
jan_month_str = "[\d]?[\s]?[jJ]an[\w]*[,-]?[\s]?[\d]?"
feb_month_str = "[\d]?[\s]?[fF]eb[\w]*[,-]?[\s]?[\d]?"
mar_month_str = "[\d]?[\s]?[M]ar(?:ch)*[,-]?[\s]?[\d]?"
apr_month_str = "[\d]?[\s]?[aA]pr[\w]*[,-]?[\s]?[\d]?"
may_month_str = "[\d]?[\s]?[M]ay[,-]?[\s]?[\d]?"
jun_month_str = "[\d]?[\s]?[jJ]un[e]*[,-]?[\s]?[\d]?"
jul_month_str = "[\d]?[\s]?[jJ]ul[\w]*[,-]?[\s]?[\d]?"
aug_month_str = "[\d]?[\s]?[aA]ug[\w]*[,-]?[\s]?[\d]?"
sep_month_str = "[\d]?[\s]?[sS]ep[tember]*[,-]?[\s]?[\d]?"
oct_month_str = "[\d]?[\s]?[oO]ct[\w]*[,-]?[\s]?[\d]?"
nov_month_str = "[\d]?[\s]?[nN]ov[ember]*[,-]?[\s]?[\d]?"
dec_month_str = "[\d]?[\s]?[dD]ec[ember]*[,-]?[\s]?[\d]?"

new_menu_str_cleaned = re.sub(jan_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(feb_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(mar_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(apr_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(may_month_str, " ", new_menu_str_cleaned)
new_menu_str_cleaned = re.sub(jun_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(jul_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(aug_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(sep_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(oct_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(nov_month_str, " ", new_menu_str_cleaned, flags = re.I)
new_menu_str_cleaned = re.sub(dec_month_str, " ", new_menu_str_cleaned, flags = re.I)


#===   remove stopwords
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english")) # build the stopwords set

new_menu_str_cleaned_recreated = []
for i in range(len(cleaned_tokens)-1):
	if cleaned_tokens[i][0] not in eng_stopwords:
		new_menu_str_cleaned_recreated.append(cleaned_tokens[i][0])

#type(new_menu_str_cleaned_recreated)

# identify the words that have at least N consecutive uppercase characters
# =============================================================================
# set(re.findall("[A-Z]{2,}", new_menu_str_cleaned))
# set(re.findall("[A-Z]{3,}", new_menu_str_cleaned))
# set(re.findall("[A-Z]{4,}", new_menu_str_cleaned))
# =============================================================================


#==============================================================================
#===   tokenization
#==============================================================================
#===   create the document set of word tokens, and clean up the tokens for the "Dictionary" function later
from nltk.tokenize import RegexpTokenizer, word_tokenize
tokenizer = RegexpTokenizer('[a-zA-Z]{3,}')  # taking out words of length at least 3
words_list = tokenizer.tokenize(str(new_menu_str_cleaned_recreated))
words_list = [word.lower() for word in words_list]

words_combined = []
for i, word in enumerate(words_list):
	words_combined.append(word)
		
len(set(words_combined))  # there are 43164 unique words


#==============================================================================
#===   perform lemmatization 
#==============================================================================
from nltk.stem import WordNetLemmatizer
WNlemma = WordNetLemmatizer()

words_combined_lemmatized = []
for text in set(words_combined):
		words_combined_lemmatized.append(WNlemma.lemmatize(text, "v"))

len(set(words_combined_lemmatized))  # after lemmatization, we get 37570 unique words

#==============================================================================
#===   perform stemming to get root words
#==============================================================================
# =============================================================================
# from nltk.stem import PorterStemmer
# port = PorterStemmer()
# 
# words_combined_lemmatized_stemmed = []
# for text in words_combined_lemmatized:
# 	words_combined_lemmatized_stemmed.append(port.stem(text))
# 
# len(set(words_combined_lemmatized_stemmed)) # after stemming, we get 6686 unique words
# 
# =============================================================================
word_vector = words_combined_lemmatized
#word_vector = words_combined_lemmatized_stemmed

# prepare to check against the correctly spelled words
from nltk.corpus import words
correct_spellings = words.words()

# =============================================================================
# words_filtered = []
# for text in word_vector:
# 	if text in correct_spellings:
# 		words_filtered.append(text)
# =============================================================================

words_filtered = []
for text in word_vector:
	words_filtered.append(text)

len(set(words_filtered))  # we have gotten 37570 words (if lemmatize only, else 2657), out of the total number of 83309 words


# #==============================================================================
# #===   Perform LDA topic modelling
# #==============================================================================
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# remove tokens that don't appear in at least 1 document
# remove tokens that appear in more than 20% of documents
vect = CountVectorizer(min_df = 5, max_df = 0.2, stop_words = 'english')

# fit and transform
vect.fit(words_filtered)
X = vect.transform(words_filtered)

# convert sparse matrix to gensim corpus
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns = False)

# mapping from word IDs to words
id_map = dict((v,k) for k, v in vect.vocabulary_.items())

# initialise the LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word = id_map,
										   passes = 25, random_state = 34)

# print out the LDA topics
ldamodel.print_topics(num_topics = 10, num_words = 10)

#==============================================================================
#===   Perform text categorisation
#==============================================================================
# take a look at a new document
new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]

    
X_test_vectorized = vect.transform(new_doc)
corpus_test = gensim.matutils.Sparse2Corpus(X_test_vectorized, documents_columns = False)    
list(ldamodel.get_document_topics(corpus_test))[0]


# #==============================================================================
# #===   Perform LDA using bag of words
# #==============================================================================		
# =============================================================================
# from gensim import corpora, models
# dictionary = corpora.Dictionary([words_filtered])  # create dictionary
# bow_corpus = [dictionary.doc2bow(doc) for doc in [words_filtered]] # create the corpus / bag-of-words representation
# 
# ldamodel_bow = gensim.models.LdaMulticore(bow_corpus, num_topics = 10, 
# 									  id2word = dictionary, passes = 2, workers = 2)
# 
# ldamodel_bow.print_topics(num_topics = 5, num_words = 10)
# 
# for idx, topic in ldamodel_bow.print_topics(-1):
# 	print('Topic: {} \nWords: {}'.format(idx, topic))
# =============================================================================


# #==============================================================================
# #===   Perform LDA using TFIDF (some issues)
# #==============================================================================		
# =============================================================================
# tfidf = models.TfidfModel(bow_corpus)
# corpus_tfidf = tfidf[bow_corpus]
# 
# ldamodel_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics = 10,
# 											id2word = dictionary, passes = 2, workers = 4)
# 
# for idx, topic in ldamodel_tfidf.print_topics(-1):
# 	print('Topic: {} Word: {}'.format(idx, topic))
# =============================================================================














