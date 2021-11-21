import requests   
from bs4 import BeautifulSoup as bs 
import re 
import pandas as pd

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty reviews list 
rvs=[]


for i in range(1,20):
  ip=[]  
  url="https://www.thehindu.com/"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")
  reviews = soup.find_all("a")
  for i in range(len(reviews)):
    ip.append(reviews[i].text)  
 
  rvs=rvs+ip  

#since more data printing only the recent headlines

revs=[x for x in rvs if (len(x)>18 and x != " ")]
word=['\n\n','\n','\t\t','\t']
revs=pd.DataFrame(revs,columns=["Text"] )
revs=revs.replace(word,"",regex=True)
print(revs[0:10])

# writng reviews in a text file 
with open("antiglare.txt","w",encoding='utf8') as output:
    output.write(str(rvs))
	

# Joinining all the reviews into single paragraph 
revstring = " ".join(rvs)

import nltk
#removing unwanted symbols
revstring = re.sub("[^A-Za-z" "]+"," ", revstring).lower()
revstring = re.sub("[0-9" "]+"," ", revstring)

# words that contained in reviews
reviews_words = revstring.split(" ")

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(reviews_words)
#you can create yuour own stopwords to remove from the extracted titles
#place names were added to the data while extracted added those to the stopword
with open("I:/Completed Assignments/Text Mining/stop.txt","r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

reviews_words = [w for w in reviews_words if not w in stop_words]

# Joinining all the reviews into single paragraph 
rev_string = " ".join(reviews_words)

# WordCloud can be performed on the string inputs.
# Corpus level word cloud

wordcloud_ip = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(rev_string)

plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("I:/Completed Assignments/Text Mining/positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)

# negative words Choose path for -ve words stored in system
with open("I:/Completed Assignments/Text Mining/negative-words.txt", "r") as neg:
  negwords = neg.read().split("\n")

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)


# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = revstring.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", w)) for w in text1]

# Create a set of stopwords
stopwords= set(STOPWORDS)

# Remove stop words
text_content = [w for w in text_content if w not in stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud

words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=set(stopwords))
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Unigram
dict1=[''.join(i) for i in nltk_tokens if not i in stopwords]
vectorizer = CountVectorizer(ngram_range=(1,1))
bag_of_words1 = vectorizer.fit_transform(dict1)
vectorizer.vocabulary_

sum_words1 = bag_of_words1.sum(axis=0)
words_freq1 = [(word, sum_words1[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq1 =sorted(words_freq1, key = lambda x: x[1], reverse=True)
print(words_freq1[:100])

# Generating wordcloud
words_dict1 = dict(words_freq1)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=set(stopwords))
wordCloud.generate_from_frequencies(words_dict1)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()