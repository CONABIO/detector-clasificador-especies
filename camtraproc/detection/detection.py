import tensorflow as tf
import pandas as pd
import numpy as np
from camtraproc.settings import DETECTOR_BATCH_SIZE, DETECTOR_THRESHOLD
from camtraproc.utils.utils import get_url_s3
import PIL.Image
import cv2
import os


def load_model(checkpoint):
    """
    Load a detection model (i.e., create a graph) from a .pb file
    """

    print('Creating Graph...')
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print('...done')

    return graph

def get_species_and_humans_indexes(detection_classes,scores,boxes,threshold):
    humans = [(f,boxes[f][[g for g in (scores[f] >= threshold).nonzero()[0] if g in (detection_classes[f] == 2).nonzero()[0]] ],
               scores[f][[g for g in (scores[f] >= threshold).nonzero()[0] if g in (detection_classes[f] == 2).nonzero()[0]] ]) 
               for f in range(len(scores)) if (detection_classes[f][scores[f] >= threshold] == 2).any() ]
#    humans_scores = [scores[f] for f in range(len(scores)) if (detection_classes[f][scores[f] >= threshold] == 2).any()]
    maybe_humans = [(f,boxes[f][[g for g in ((scores[f] > 0.2) & ~(scores[f] >= threshold)).nonzero()[0] if g in (detection_classes[f] == 2).nonzero()[0]] ],
                     scores[f][[g for g in ((scores[f] > 0.2) & ~(scores[f] >= threshold)).nonzero()[0] if g in (detection_classes[f] == 2).nonzero()[0]] ])
                     for f in range(len(scores)) if (detection_classes[f][scores[f] > 0.2] == 2).any() and not (detection_classes[f][scores[f] >= threshold] == 2).any() ]
#    maybe_humans_scores = [scores[f]  for f in range(len(scores)) if (detection_classes[f][scores[f] > 0.2] == 2).any() and not (detection_classes[f][scores[f] >= threshold] == 2).any() ]
    species = [(f, boxes[f][[g for g in (scores[f] >= threshold).nonzero()[0] if g in (detection_classes[f] == 1).nonzero()[0]]], 
                scores[f][[g for g in (scores[f] >= threshold).nonzero()[0] if g in (detection_classes[f] == 1).nonzero()[0]]])
                for f in range(len(scores)) if (detection_classes[f][scores[f] >= threshold] == 1).any() and f not in humans]
    return species, humans, maybe_humans

def get_dataframe(df,ind_bboxes):
    score = []
    x = []
    y = []
    w = []
    h = []
    file_names = []
    latlong = []
    index = []
    index_batch = []
    date = []
    frame_rate = [] 
    item_type = []
    date_delta = []
    sequence_id = []
    index1 = []
    num_frame = []

    if type(df['item_file']) == str:
        for b in range(len(ind_bboxes)):
            for i in range(ind_bboxes[b][1].shape[0]):
                score.append(ind_bboxes[b][2][0])
                x.append(ind_bboxes[b][1][i,1])
                y.append(ind_bboxes[b][1][i,0])
                w.append(ind_bboxes[b][1][i,3]-ind_bboxes[b][1][i,1])
                h.append(ind_bboxes[b][1][i,2]-ind_bboxes[b][1][i,0])
                file_names.append(df['item_file'].split('.')[0] + '_bb' + str(i) + '.' + df['item_file'].split('.')[1])
                latlong.append(df['latlong'])
                index.append(df['id'])
                try:
                    date.append(df['date'])
                    frame_rate.append(df['frame_rate'])
                    item_type.append(df['item_type'])
                    index1.append(df['index1'])
                    date_delta.append(df['date_delta'])
                    sequence_id.append(df['sequence_id'])
                    num_frame.append(df['num_frame'])
                except:
                    pass
                index_batch.append(ind_bboxes[b][0])
    else:
        for b in range(len(ind_bboxes)):
            for i in range(ind_bboxes[b][1].shape[0]):
                score.append(ind_bboxes[b][2][0])
                x.append(ind_bboxes[b][1][i,1])
                y.append(ind_bboxes[b][1][i,0])
                w.append(ind_bboxes[b][1][i,3]-ind_bboxes[b][1][i,1])
                h.append(ind_bboxes[b][1][i,2]-ind_bboxes[b][1][i,0])
                file_names.append(df['item_file'][ind_bboxes[b][0]].split('.')[0] + '_bb' + str(i) + '.' + df['item_file'][ind_bboxes[b][0]].split('.')[1])
                latlong.append(df['latlong'][ind_bboxes[b][0]])
                index.append(df['id'][ind_bboxes[b][0]])
                try:
                    date.append(df['date'])
                    frame_rate.append(df['frame_rate'])
                    item_type.append(df['item_type'])
                    index1.append(df['index1'])
                    date_delta.append(df['date_delta'])
                    sequence_id.append(df['sequence_id'])
                    num_frame.append(df['num_frame'])
                except:
                    pass
                index_batch.append(ind_bboxes[b][0])
    ndf = pd.DataFrame(index, columns=['id'])
    ndf['item_file'] = file_names
    ndf['index_batch'] = index_batch
    ndf['x'] = x
    ndf['y'] = y
    ndf['w'] = w
    ndf['h'] = h
    ndf['score'] = score
    ndf['latlong'] = latlong
    try:
        ndf['date'] = date
        ndf['frame_rate'] = frame_rate
        ndf['item_type'] = item_type
        ndf['index1'] = index1
        ndf['date_delta'] = date_delta
        ndf['sequence_id'] = sequence_id
        ndf['num_frame'] = num_frame
    except:
        pass

    return ndf

def crop_generator(img,x,y,w,h):
    interim_size_x = img.shape[1]
    interim_size_y = img.shape[0]
    #1.6 is the first crop and 0.8 is the second crop = 1.2800000000000002
    l = (np.minimum(np.maximum(w*1.6*0.8,h*1.6*0.8), np.minimum(interim_size_x - 1,int((1 - .04) * interim_size_y) - 1))).astype(int)
    ofsx = int((l - w)/2)
    ofsy = int((l - h)/2)
    x1 = np.maximum(0, x - ofsx)
    y1 = np.maximum(0, y - ofsy)
    x2 = np.minimum(interim_size_x - 1, x + w + ofsx)
    y2 = np.minimum(np.maximum(int((1 - .04) * interim_size_y), y + h), y + h + ofsy)

    def crop_images(x1, y1, x2, y2, x, y, w, h, ofsx, ofsy, image_x):
        crop = image_x[y1:y2, x1:x2, :]
        if x2-x1 < 2*ofsx + w:
            if x - x1 < ofsx:
                left = ofsx - (x - x1)
                right = 0
                crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_WRAP)
            if x2 - x - w < ofsx:
                left = 0
                right = ofsx - (x2 - x - w)
                crop = cv2.copyMakeBorder(crop, 0, 0, left, right, cv2.BORDER_WRAP)
        if y2-y1 < 2*ofsy + h:
            if y - y1 < ofsy:
                top = ofsy - (y - y1)
                bottom = 0
                crop = cv2.copyMakeBorder(crop, top, bottom, 0, 0, cv2.BORDER_WRAP)
            if y2 - y - h < ofsy:
                top = 0
                bottom = ofsy - (y2 - y - h)
                crop = cv2.copyMakeBorder(crop, top, bottom, 0, 0, cv2.BORDER_WRAP)

        im = crop
#        im = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR) #Image.fromarray(crop)

        return im

    new_x = crop_images(x1, y1, x2, y2, x, y, w, h, ofsx, ofsy, img)

    return new_x



def generate_detections(detection_graph,images,bboxes_dir,df):
    """
    boxes,scores,classes,images = generate_detections(detection_graph,images)

    Run an already-loaded detector network on a set of images.

    [images] can be a list of numpy arrays or a list of filenames.  Non-list inputs will be
    wrapped into a list.

    Boxes are returned in relative coordinates as (top, left, bottom, right);
    x,y origin is the upper-left.

    [boxes] will be returned as a numpy array of size nImages x nDetections x 4.

    [scores] and [classes] will each be returned as a numpy array of size nImages x nDetections.

    [images] is a set of numpy arrays corresponding to the input parameter [images], which may have
    have been either arrays or filenames.
    """
    boxes = []
    scores = []
    classes = []

    nImages = len(images)
    num = nImages // DETECTOR_BATCH_SIZE
    if nImages % DETECTOR_BATCH_SIZE:
        num = num + 1
#    images_st = [images[f*os.getenv('DETECTOR_BATCH_SIZE'):(f+1)*os.getenv('DETECTOR_BATCH_SIZE')] for f in range(num)]
#    if nImages % os.getenv('DETECTOR_BATCH_SIZE'):
#        images_st.append(images[num*os.getenv('DETECTOR_BATCH_SIZE'):])

    with detection_graph.as_default():

        with tf.Session(graph=detection_graph) as sess:

            for inu in range(num):

                print('Processing images {} of {}'.format(inu,num))
                if nImages == 1:
                    images_expanded = imageNP_expanded = np.expand_dims(images[0], axis=0)
                elif nImages % DETECTOR_BATCH_SIZE and inu == num - 1:
                    images_expanded = np.stack(images[inu*DETECTOR_BATCH_SIZE:], axis=0)
                else:
                    images_expanded = np.stack(images[inu*DETECTOR_BATCH_SIZE:(inu+1)*DETECTOR_BATCH_SIZE], axis=0) #imageNP_expanded = np.expand_dims(imageNP, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                box = detection_graph.get_tensor_by_name('detection_boxes:0')
                score = detection_graph.get_tensor_by_name('detection_scores:0')
                clss = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection
                (box, score, clss, num_detections) = sess.run(
                        [box, score, clss, num_detections],
                        feed_dict={image_tensor: images_expanded})

                boxes.append(box)
                scores.append(score)
                classes.append(clss)

    # Squeeze out the empty axis
    boxes = np.concatenate(boxes,axis=0)
    scores = np.concatenate(scores,axis=0)
    classes = np.concatenate(classes,axis=0).astype(int)


    species, humans, maybe_humans = get_species_and_humans_indexes(classes,scores,boxes,DETECTOR_THRESHOLD)

    species_df = get_dataframe(df,species)
    for r in np.array(species_df):
        x = int(r[3]*images[r[2]].shape[1] + 0.5)
        y = int(r[4]*images[r[2]].shape[0] + 0.5)
        w = int(r[5]*images[r[2]].shape[1] + 0.5)
        h = int(r[6]*images[r[2]].shape[0] + 0.5)
        image_to_write = cv2.cvtColor(crop_generator(images[r[2]],x,y,w,h), cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(os.path.join(bboxes_dir, r[1])), exist_ok=True)
        cv2.imwrite(os.path.join(bboxes_dir, r[1]), image_to_write)

#    humans_df = df[df['ind'].isin(humans)]
#    maybe_humans_df = df[df['ind'].isin(maybe_humans)]
    humans_df = get_dataframe(df,humans)
    maybe_humans_df = get_dataframe(df,maybe_humans)

    return species_df, humans_df, maybe_humans_df

def get_frames(key,df,mode,bucket=None):
    def get_df(df,f):
        df1 = df.copy()
        df1['item_file'] = os.path.basename(df['item_file']).split('.')[0] + '_{}.JPG'.format(f)
        df1['num_frame'] = f
        return df1
    if mode == 's3':
        url = get_url_s3(bucket,key)
    else:
        url = df['item_file']
    cap = cv2.VideoCapture(url)
    frames = []
    no_frames = []
    while True:
        status, frame = cap.read()
        if not status:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    nframes = len(frames)
    ndf = pd.concat([get_df(df,f) for f in range(len(frames))])
    return frames, ndf

def get_images(filepath,df,dirpath=''):
#    def get_df(df,f=0):
#        df1 = df.copy()
#        df1['item_file'] = df['item_file'].split('.')[0] + '_{}.'.format(f) + df['item_file'].split('.')[1]
#        return df1
    try:
        img = PIL.Image.open(os.path.join(dirpath, filepath)).convert("RGB")
        img = [np.array(img)]
        df1 = df.copy()
        df1['num_frame'] = 0
        return img, df
    except Exception as e:
        df1 = df.copy()
        df1['num_frame'] = -1
        return False, df

def run_megadetector(detection_graph4,bboxes_dir,df,media_type,mode,bucket=None):
    species = []
    humans = []
    maybe_humans = []

    if media_type == 'video':
        for i in range(len(df)):
            images,ndf = get_frames(df.iloc[i]['item_file'],df.iloc[i],mode,bucket)
            print('video_{}/{}'.format(i,len(df)))
            species_df, humans_df, maybe_humans_df = generate_detections(detection_graph4,images,bboxes_dir,ndf)
            species.append(species_df)
            humans.append(humans_df)
            maybe_humans.append(maybe_humans_df)
    elif media_type == 'image':
        for i in range(len(df)):
            images,ndf = get_images(df.iloc[i]['item_file'],df.iloc[i])
            if images is not False:
                species_df, humans_df, maybe_humans_df = generate_detections(detection_graph4,images,bboxes_dir,ndf)
                species.append(species_df)
                humans.append(humans_df)
                maybe_humans.append(maybe_humans_df)
    else:
        raise Exception('media type is invalid!')
        
    dfspecies = pd.concat(species)
    dfhumans = pd.concat(humans)
    dfmaybe_humans = pd.concat(maybe_humans)
    return [dfspecies,dfhumans,dfmaybe_humans]

