import swifter
import requests
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from ast import literal_eval
from longformer_model.prediction import predict
# from prediction import predict

# df = pd.read_csv('arti_scrap_v1.csv', sep='\t')
df = pd.read_csv('/home/tosi-n/Credibron/filter_art/arti_scrap_v3.csv', sep='\t')#, nrows=5000

print(len(df.index))

print(list(df.columns.values))

source_cred_endpoint = "http://34.102.156.66/source_credibility"


def get_source_credibility(url):
    # text = "this text is biased"
    body = {
        "_id": 0,
        "url": url
    }
    try:
        response = requests.post(source_cred_endpoint, json=body)
        score = 0
        if response.status_code == 200:
            label = response.json()['credibility_label']
    except:
        pass
    return label

df_ = pd.DataFrame( df['body'].astype('str').progress_apply(get_source_credibility).to_list(), columns =['source_cred']) 
df = pd.concat([df, df_], axis=1)


# sentiment = "http://35.231.199.161:8008/v1/predict/longformer_cls"


# def get_sentiment(url):
#     # text = "this text is biased"
#     body = {
#         "_id": 0,
#         "url": url
#     }
#     try:
#         response = requests.post(sentiment, json=body)
#         score = 0
#         if response.status_code == 200:
#             label = response.json()['credibility_label']
#     except:
#         pass
#     return label

# df_ = pd.DataFrame( df['body'].astype('str').progress_apply(get_sentiment).to_list(), columns =['avg_sentiment']) 
# df = pd.concat([df, df_], axis=1)


df['concat_features'] = df.title.str.cat(df.source_cred,sep=" ")
# df['concat_features'] = df.title.str.cat(df.avg_sentiment,sep=" ")

# df['concat_features'] = df.concat_features.str.cat(df.source_cred,sep=" ")

df['concat_features'] = df.concat_features.str.cat(df.text,sep=" ")


# df.rename(columns={'text': 'text_',
#                         'concat_features': 'text'},
#                     inplace=True)

df = df[['title', 'avg_sentiment', 'source_cred', 'text']]
print(len(df.index))

df['article_id'] = [x for x in range(1, len(df.values)+1)] 
df = df[['article_id', 'title', 'text_avg_sent', 'source_cred', 'text']]




# article_cred_endpoint = "http://35.231.199.161:8008/v1/predict/longformer_cls"
article_cred_endpoint = "http://34.107.167.39/v2/predict/longformer_cls"


def get_article_credibility_score(_id, title, sentiment, source_cred, body):

    # sentiment = json.loads(sentiment)
    sentiment = literal_eval(sentiment)
    body_data = { "_id" : _id, "title" : title, "sentiment" : sentiment, "source_cred" : source_cred, "body" : body }
    print(type(sentiment))
    print(sentiment)
    try:
        response = requests.post(article_cred_endpoint, json=body_data)
        if response.status_code == 200:
            cred = response.json()
            # print(cred)
            label,confid_score = cred['predictions'], cred['confidence']
            # latency = response.elapsed.total_seconds()
            # print(latency)
    except:
        pass
    return label, confid_score

df_ = pd.DataFrame(df.progress_apply(lambda row: get_article_credibility_score(row['article_id'], row['title'], row['text_avg_sent'], row['source_cred'], row['text']), axis=1).to_list(), columns =['class', 'confidence'])

df = pd.concat([df, df_], axis=1)

# df_ = pd.DataFrame( df['body'].astype('str').progress_apply(get_article_credibility_score).to_list(), columns =['class', 'confidence']) 
# df = pd.concat([df, df_], axis=1)

df['confidence'] = df['confidence'].astype(float) * 100
# df.drop(df[df['confidence'] >= 50].index, inplace = True)
df.drop(df[df['confidence'] <= 50].index, inplace = True)
print(len(df.index))

df.to_csv('low_conf.csv',index=False, sep='\t')