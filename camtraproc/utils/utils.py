import pandas as pd
from datetime import datetime, timedelta
from camtraproc.settings import AWS_SERVER_PUBLIC_KEY, AWS_SERVER_SECRET_KEY, ITEM_TYPE, ITEM_LIST_ENDPOINT, USER, PSSWD,WITH_MOTION_SEQ, SEQ_TIME_DELTA
import boto3
import json
import requests
import re
import os

def get_url_s3(bucket,key,region='us-west-2'):
    s3_client = boto3.client('s3',region_name=region, aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
                         aws_secret_access_key=AWS_SERVER_SECRET_KEY)
    return s3_client.generate_presigned_url( ClientMethod='get_object', Params={ 'Bucket': bucket, 'Key': key } )


def upload_s3(bucket,path,out_path,region='us-west-2'):
    s3_client = boto3.client('s3',region_name=region, aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
                         aws_secret_access_key=AWS_SERVER_SECRET_KEY)
    try:
        filename = os.path.join(path,out_path)
        object_name = filename
        response = s3_client.upload_file(filename, bucket, object_name)
    except Exception as e:
        print(e)
        return False
    return True

def get_sequence(delta):
    res = []
    td = timedelta(seconds=SEQ_TIME_DELTA) #20
    g = 1
    res.append(str(1))
    for i in range(1, len(delta)):
        if (delta[i] > td):
            g = g + 1
            res.append(str(g))
        else:
            res.append(str(g))
    return res

def query_s3(bucket,path,pattern,out_path,region='us-west-2'):
    s3 = boto3.resource('s3',region_name=region, aws_access_key_id=AWS_SERVER_PUBLIC_KEY,
                         aws_secret_access_key=AWS_SERVER_SECRET_KEY)
    my_bucket = s3.Bucket(bucket)
    obj_list = my_bucket.objects.filter(Prefix=path)
    out = [x.key for x in obj_list]
    if pattern is not None:
        pattern = re.compile(pattern)
        out = ['https://snmb.s3-us-west-2.amazonaws.com/' + x for x in out if pattern.search(x)]
        coo = ['' for f in range(len(out)) ]
        df = pd.DataFrame([f for f in range(len(out))], columns=['id'])
        df['item_file'] = out
        df['latlong'] = coo
        df.to_csv(os.path.join(out_path,'query_s3.csv'), index=False)
    return df

def query_irekua(out_path):
    filters = {
    'item_type':ITEM_TYPE
    }
    data = requests.get(ITEM_LIST_ENDPOINT, auth=(USER, PSSWD), params=filters).json()
    results = data['results']
    ep = data['next']
    while ep:
        data = requests.get(ep,auth=(USER, PSSWD)).json()
        results = results + data['results']
        ep = data['next']
    
    df = pd.DataFrame(results)[['id','item_file','collection_site']]
    df['site_id'] = df.apply(lambda x: x.collection_site['id'], axis=1)
    df['site_url'] = df.apply(lambda x: x.collection_site['url'], axis=1)
    df2 = df[['site_id','site_url']].drop_duplicates()
    df2['site'] = df2.apply(lambda x: requests.get(x.site_url,auth=(USER, PSSWD)).json(), axis=1)
    df2['latlong'] = df2.apply(lambda x: '{:.{n}f}'.format(x.site[0]['site']['geometry']['coordinates'][1],n=7) +
                           '|' + '{:.{n}f}'.format(x.site[0]['site']['geometry']['coordinates'][0],n=7), axis=1)
    df2 = df2[['site_id','latlong']]
    df = df.merge(df2,how='left',on='site_id')
    df = df[['id','item_file','latlong']]
    df.to_csv(os.path.join(out_path,'query_irekua.csv'), index=False)
    with open(os.path.join(out_path,'query_irekua.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return df

def query_manual(out_path):
    if WITH_MOTION_SEQ:
        df = pd.read_csv(os.path.join(out_path,'total_files_coor.csv')).sort_values(by='date') #id, item_file, date, latlong, frame_rate, item_type
        df = df.reset_index(drop=True)
        df['index1'] = df.index
        df['date'] = df.apply(lambda x: datetime.strptime(x.date, '%Y-%m-%d %H:%M:%S'), axis=1)
        df['date_delta'] = df['date'].diff()
        df['sequence_id'] = get_sequence(df['date_delta'])
        return df
    else:
        return pd.read_csv(os.path.join(out_path,'total_files_coor.csv')) #id, item_file, latlong
