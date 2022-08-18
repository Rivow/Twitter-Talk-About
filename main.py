import concurrent.futures
import streamlit as st
from datetime import datetime
from datetime import timedelta
import pandas as pd
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import twint


def submit():
    """This method is called when the button is clicked To apply the functions and get the output"""
    if st.session_state.topic == '':
        return
    day = datetime.combine(st.session_state.date_select, datetime.min.time())
    #print(day)
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #    times = [day + timedelta(minutes=h * 30) for h in range(48)]
    #    results = executor.map(get_tweets, times)

    #tweets = next(results)
    #for r in results:
    #    tweets = pd.concat([tweets, r])
    #tweets.drop_duplicates(subset=['id', 'tweet'], inplace=True)

    tweets = get_tweets(day)
    print(tweets.shape)
    if not tweets.empty:
        processed = prepare_tweets(tweets)
        lda_model = topic_model(processed)
        topics = clean_topics(lda_model)
        topics = pd.DataFrame.from_dict(topics)
        topics
        return topics


def get_tweets(date):
    try:
        if not topic:
            return pd.DataFrame()
        next_day = date + timedelta(minutes=30)
        c = twint.Config()
        c.Search = topic
        c.Limit = 100
        c.Since = str(date)
        c.Until = str(next_day)
        c.Pandas = True
        c.Hide_output = True
        twint.run.Search(c)
        df = twint.storage.panda.Tweets_df
        df = df[df['language'] =='en']
        df['date'] = df.date.apply(lambda x: x.split(' ')[0])
        df = df[df['date'] == str(date)]
        return df

    except Exception:
        pass


def prepare_tweets(df):
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'http\S+', '', x))
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    processed = df['tweet'].map(preprocess)
    return processed


def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        # We accept the word max since it is used for driver Max Verstappen
        if (token not in gensim.parsing.preprocessing.STOPWORDS and len(
                token) > 3):
            result.append(lemmatize_stemming(token))

    return result


def topic_model(processed):
    dictionary = gensim.corpora.Dictionary(processed)

    # filter words that abear in less than 20% off tweets and keep all popular words
    dictionary.filter_extremes(no_below=0.2, no_above=0.45, keep_n=100000)

    # Create a Bag-of-Words model fot every tweet
    bow_corpus = [dictionary.doc2bow(tweet) for tweet in processed]

    # creating a TF_IDF model
    tfidf = models.TfidfModel(bow_corpus)

    # Apply transformation to the entire corpus call
    corpus_tfidf = tfidf[bow_corpus]

    # LDA modeling
    lda_model = gensim.models.LdaModel(corpus_tfidf,
                                          num_topics=3,
                                          id2word=dictionary,
                                          passes=2)
    return lda_model


def clean_topics(model):
    topic_dict = {}
    for idx, topic in model.print_topics(-1):
        topic_list = []
        words = topic.split('+ ')
        for word in words:
            topic_list.append(word[7:-2])
        topic_dict[idx] = topic_list
    return topic_dict




if __name__ == '__main__':
    min_date = datetime.strptime('2006-07-15', '%Y-%m-%d')
    max_date = datetime.today().date()
    st.text_input('Input Topic', key='topic')
    topic = st.session_state.topic
    st.date_input('Input Date', key='date_select', min_value=min_date, max_value=max_date)
    st.button('Submit', key='submit', on_click=submit())

