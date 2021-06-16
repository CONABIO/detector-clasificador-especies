import pandas as pd
from camtraproc.settings import AWS_SERVER_PUBLIC_KEY, AWS_SERVER_SECRET_KEY, ITEM_LIST_ENDPOINT, USER, PSSWD
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
    data = requests.get(ITEM_LIST_ENDPOINT, auth=(USER, PSSWD)).json()
    df = pd.DataFrame(data['results'])[['id','item_file']]
    df['latlong'] = ['19.0650555556|-96.909' for f in range(len(df))]
    df.to_csv(os.path.join(out_path,'query_irekua.csv'), index=False)
    with open(os.path.join(out_path,'query_irekua.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return df

def query_manual(out_path):
    return pd.read_csv(os.path.join(out_path,'total_files_coor.csv'))

