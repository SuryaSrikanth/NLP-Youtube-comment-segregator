videoID = 'hA6hldpSTF8'
key = 'AIzaSyDX9PwoPwIu9698S33C2aGUxYC8usPh8vI'
number_of_topics = 10
passes = 5


import pandas as pd 
import os
import googleapiclient.discovery
import pickle
import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
import nltk
import gensim
# nltk.download('wordnet')
# nltk.download('punkt')

def getComments(videoId, nextPageToken = None):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = key

    youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey = DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="snippet, replies",
        videoId = videoId,
        maxResults = 100,
        pageToken = nextPageToken
    )
    response = request.execute()
    return response

def preprocess(text):
    t = " ".join(re.findall("[a-zA-Z]+", text))
    t = t.split(" ")
    result = []
    for token in t:
        token = token.strip('.?,!<>:;[]{}!-"')
        if token not in stopwords.words('english') and len(token) >= 3:
            result.append(stemmer.stem(lemmatizer.lemmatize(token, pos='v')))
    return result

def get_topic(L):
    return max(L, key=lambda lis: lis[1])



if __name__ == "__main__":
    commentListResponce = getComments(videoID)

    comments = [comment['snippet']['topLevelComment']['snippet']['textOriginal'] for comment in commentListResponce['items']]


    model_file = 'finalized_spam_model.sav'
    vec_file = 'vectorizer.sav'
    spam_model = pickle.load(open(model_file,'rb'))
    vect = pickle.load(open(vec_file,'rb'))


    data = pd.DataFrame(comments,columns=['comments'])
    small_comments = []

    for i in range(len(data['comments'])):
        comment = " ".join(re.findall("[a-zA-Z]+", data['comments'][i]))
        if (len(nltk.word_tokenize(comment))<3):
            small_comments.append(i)

    data.drop(small_comments, inplace=True)

    for index in sorted(small_comments, reverse=True):
        del comments[index]
        
    data.reset_index(drop=True, inplace=True)

    comments_vec = vect.transform(data['comments'])

    spam_predictions = spam_model.predict(comments_vec)

    filtered = []
    for i in range(len(spam_predictions)):
        if (spam_predictions[i] == 0) and (len(data['comments'][i]) != 0):
            filtered.append(data['comments'][i])
    filtered_data = pd.DataFrame(filtered,columns=['comments'])   

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()


    filtered_data['processed'] = filtered_data.apply(lambda row: preprocess(row['comments']),axis=1)


    df = pd.DataFrame()
    df['text'] = filtered_data['processed']
    df = df[df['text'].map(lambda x: len(x)) > 0]


    dictionary = gensim.corpora.Dictionary(df['text'])
    bag_of_words = [dictionary.doc2bow(x) for x in df['text']]
    tfidf = gensim.models.TfidfModel(bag_of_words)
    tfidf_corpus = tfidf[bag_of_words]
    lda_model_tfidf = gensim.models.LdaMulticore(tfidf_corpus,num_topics = number_of_topics, id2word = dictionary, passes=passes, workers = 2)


    df['bow'] = df.apply(lambda x: dictionary.doc2bow(x['text']), axis =1)
    df['topic_scores'] = df.apply(lambda x: lda_model_tfidf[x['bow']],axis=1)


    df['topic'] = df.apply(lambda x: get_topic(x['topic_scores'])[0],axis=1)


    final = pd.DataFrame()
    final['comments'],final['processed'],final['topic'] = filtered_data['comments'],filtered_data['processed'],df['topic']



    DIC = {}
    DIC_topics = {}



    for i in range(number_of_topics):
        txt_topics = ' '.join(final[final['topic'] == i]['comments'])
        DIC[i] = []
        DIC_topics[i] = txt_topics

    for i in range(len(final['comments'])):
        DIC[final['topic'][i]].append(final['comments'][i])

    for i in DIC_topics.keys():
        DIC_topics[i] = " ".join(re.findall("[a-zA-Z]+", DIC_topics[i]))

    from gensim.summarization import keywords
    for i in DIC_topics.keys():
        DIC_topics[i] = keywords(DIC_topics[i]).splitlines()
        if len(DIC_topics[i]) == 0:
            DIC_topics[i] = ['Others']


    print('\n-----------------------------------------------------\n')
    print(DIC)
    print('\n-----------------------------------------------------\n')
    print(DIC_topics)        
    print('\n-----------------------------------------------------\n')
