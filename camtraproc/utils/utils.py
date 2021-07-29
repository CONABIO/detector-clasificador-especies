import pandas as pd
from datetime import datetime, timedelta
from camtraproc.settings import AWS_SERVER_PUBLIC_KEY, AWS_SERVER_SECRET_KEY, ITEM_TYPE, ITEM_LIST_ENDPOINT, MRUN_LIST_ENDPOINT, MPRED_ENDPOINT, USER, PSSWD,WITH_MOTION_SEQ, SEQ_TIME_DELTA, MODEL_VERSION, MODEL_RUN, ANNOT_TYPE, EVENT_TYPE_FAUNA, EVENT_TYPE_ANTROP, LABEL, IMAGE_TYPE, VIDEO_TYPE
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

def post_predictions(df,is_fauna):
#    data = {
#    "item": '14',
#    "annotation_type": '1',
#    "event_type": '1',
#    "annotation": '[0.2,0.2,0.1,0.1]',
#    "labels": ['2'],
#    "model_version": '1',
#    "model_run": '1',
#    "certainty": 0.96
#}
    try:
        df['certainty'] = df['score']
        df['annotation_metadata'] = df['num_frame']
        df['item'] = df.apply(lambda x: str(int(x.id)), axis=1)
        df['annotation'] = df.apply(lambda r: str([r.x,r.y,r.w,r.h]), axis=1)
        df = df[['item','annotation','certainty','annotation_metadata']]
        if is_fauna:
            df['event_type'] = df.apply(lambda x: EVENT_TYPE_FAUNA, axis=1)
        else:
            df['event_type'] = df.apply(lambda x: EVENT_TYPE_ANTROP, axis=1)
        df['labels'] = df.apply(lambda x: [LABEL], axis=1)
        df['model_version'] = df.apply(lambda x: MODEL_VERSION, axis=1)
        df['model_run'] = df.apply(lambda x: MODEL_RUN, axis=1)
        df['annotation_type'] = df.apply(lambda x: ANNOT_TYPE, axis=1)

        dfr_list = df.to_dict('records')
    
        for dfr in dfr_list:
            print(dfr)
            r = requests.post(MPRED_ENDPOINT, auth=(USER, PSSWD), data=dfr)
            print(r)
    except Exception as e:
        print(e)
        return False
    return True

def get_date(date):
    try:
        d = datetime.strptime(date.split('.')[0], '%Y-%m-%dT%H:%M:%S')
    except Exception as e:
        print(e)
        d = None
    return d

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
    'model_version':MODEL_VERSION
    }
    
    results = None
    data = requests.get(MRUN_LIST_ENDPOINT, auth=(USER, PSSWD),params=filters).json()
    results = data['results']
    ep = data['next']
    while ep:
        data = requests.get(ep,auth=(USER, PSSWD)).json()
        results = results + data['results']
        ep = data['next']
    
    dfr = pd.DataFrame(results)
    dfr['mversion_id'] = dfr.apply(lambda x: x.model_version['id'], axis=1)
    dfr['item_id'] = dfr.apply(lambda x: x['item']['id'], axis=1)
    dfr = dfr[dfr['mversion_id'] == int(MODEL_VERSION)][['mversion_id','item_id']].drop_duplicates()
    
    filters = {
    'item_type':IMAGE_TYPE
    }
    results = []
    data = requests.get(ITEM_LIST_ENDPOINT, auth=(USER, PSSWD), params=filters).json()
    results = data['results']
    ep = data['next']
    while ep:
        data = requests.get(ep,auth=(USER, PSSWD)).json()
        results = results + data['results']
        ep = data['next']
    dfi = pd.DataFrame(results)[['id','item_file','item_type','collection_site','captured_on']]

    filters = {
    'item_type':VIDEO_TYPE
    }
    data = requests.get(ITEM_LIST_ENDPOINT, auth=(USER, PSSWD), params=filters).json()
    results = results + data['results']
    ep = data['next']
    while ep:
        data = requests.get(ep,auth=(USER, PSSWD)).json()
        results = results + data['results']
        ep = data['next']

    dfv = pd.DataFrame(results)[['id','item_file','item_type','collection_site','captured_on']]
    df = pd.concat([dfi,dfv])
    dfi = dfv = results = None
    df = df.merge(dfr,how='left',left_on='id',right_on='item_id')
    df = df[df['mversion_id'].isnull()]
    df['item_type_id'] = df.apply(lambda x: x.item_type['id'], axis=1)
    df = df.drop(['item_type'],axis=1)
    df = df.rename({'item_type_id':'item_type'}, axis=1)
    df['site_id'] = df.apply(lambda x: x.collection_site['id'], axis=1)
    df['site_url'] = df.apply(lambda x: x.collection_site['url'], axis=1)
    df2 = df[['site_id','site_url']].drop_duplicates()
    df2['site'] = df2.apply(lambda x: requests.get(x.site_url,auth=(USER, PSSWD)).json(), axis=1)
    df2['latlong'] = df2.apply(lambda x: '{:.{n}f}'.format(x.site[0]['site']['geometry']['coordinates'][1],n=7) + '|' + '{:.{n}f}'.format(x.site[0]['site']['geometry']['coordinates'][0],n=7), axis=1)
    df2 = df2[['site_id','latlong']]
    df = df.merge(df2,how='left',on='site_id')
    df = df[['id','item_file','latlong','item_type','captured_on']]
    df = df[~df['captured_on'].isnull()]
    
    df = df.sort_values(by='captured_on') #id, item_file, date, latlong, frame_rate, item_type
    df = df.reset_index(drop=True)
    df['index1'] = df.index
    df['date'] = df.apply(lambda x: get_date(x.captured_on), axis=1)
    df['date_delta'] = df['date'].diff()
    df['sequence_id'] = get_sequence(df['date_delta'])
    df = df.drop(['captured_on'],axis=1).drop_duplicates()

    df.to_csv(os.path.join(out_path,'query_irekua.csv'), index=False)
    with open(os.path.join(out_path,'query_irekua.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return df

def query_manual(out_path):
    if WITH_MOTION_SEQ:
        df = pd.read_csv(os.path.join(out_path,'total_files_coor.csv')).sort_values(by='date') #id, item_file, date, latlong, frame_rate, item_type
#        df = df.reset_index(drop=True)
#        df['index1'] = df.index
#        df['date'] = df.apply(lambda x: datetime.strptime(x.date, '%Y-%m-%d %H:%M:%S'), axis=1)
#        df['date_delta'] = df['date'].diff()
#        df['sequence_id'] = get_sequence(df['date_delta'])
        return df
    else:
        return pd.read_csv(os.path.join(out_path,'total_files_coor.csv')) #id, item_file, latlong
