import os
from pymongo import MongoClient
from tqdm import tqdm
import csv
import pandas as pd

BASE_PATH=os.path.join(os.getcwd(),'experiment/')
mongo_client = MongoClient('mongodb://{}{}:{}'.format("sina:sinapassword@", "10.142.0.26", 27017))
mongodb_c = mongo_client["zignal"]["prime_news"]

daterange = pd.date_range('2020-07-03', '2020-07-03').strftime('%Y-%m-%d')
for date_str in daterange:
    print(date_str)
    offset = 0; batch_size = 20000
    rows = list()
    while True:
        query = {"date": date_str}
        projection = {"_id": 1, "date": 1, "url": 1, "title": 1, "body": 1, "dedupe": 1}
        docs_cursor = mongodb_c.find(query, projection).skip(offset).limit(batch_size)
        docs = list(docs_cursor)
        if len(docs) == 0: break
        for qdoc in tqdm(docs):
            try:
                body = qdoc["body"].replace("\n", " ")
                dedup = qdoc.get("dedupe", {})
                row = [qdoc["_id"], qdoc["date"], qdoc["url"], qdoc['title'], body, dedup.get("isDup"), dedup.get("isMaster"), dedup.get("masterId")]
                rows.append(row)
            except Exception as ex:
                print(ex)
        offset = offset + batch_size
        if offset > 200000: break
    print(len(rows))
    cols=["article_id", "date", "url", "title", "body", "dedup_isDup", "dedup_isMaster", "dedup_masterId"] 
    df = pd.DataFrame(rows,columns=cols)
    df.to_csv(os.path.join(BASE_PATH,'{}.csv'.format(date_str)),index=False)

df = pd.read_csv('/home/tosi-n/article_credibility_api/experiment/arti_scrap_v2.csv')#, sep='\t'

print(len(df.index))