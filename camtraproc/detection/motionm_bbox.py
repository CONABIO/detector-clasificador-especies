from camtraproc.settings import IMAGE_TYPE, VIDEO_TYPE, SUBMETHOD, ACCAVG, THRESHT, MOGVAR, MOGLEARNING, MINSIZE
import numpy as np
import pandas as pd
import shapely.geometry as sg
from shapely.ops import cascaded_union
import PIL.Image
import cv2
import sys
import os

class Background:
    def __init__(self,subMethod,display_image,acc,thresh,mogvariance,media='video'):
    
        ##Subtractor Method
        self.subMethod=subMethod
        
        ####Create Background Constructor
        if self.subMethod == "Acc":
                self.running_average_image = np.float32(display_image)
                self.accAvg=acc
                self.threshT=thresh
    
        if self.subMethod == "MOG":
            #MOG method creator
            if media == 'image':
                self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False,varThreshold=float(mogvariance+20))
            if media == 'video':
                self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False,varThreshold=float(mogvariance))
            self.fgbg.setBackgroundRatio(0.95)
    
    #Frame Subtraction
    def BackGroundSub(self,camera_imageROI,learningRate):
        ## accumulated averaging
        if self.subMethod == "Acc":
            # Create an image with interactive feedback:
            self.color_image = camera_imageROI.copy()
                    
            # Smooth to get rid of false positives
            self.color_image = cv2.GaussianBlur(self.color_image,(3,3),0)
                       
            # Use the Running Average as the static background
            cv2.accumulateWeighted(self.color_image,self.running_average_image,self.accAvg)
            self.running_average_in_display_color_depth = cv2.convertScaleAbs(self.running_average_image)
            
            #Needs to be manually commented if vis
            #sourceM.displayV("Background image",10,self.running_average_in_display_color_depth)
            
            # Subtract the current frame from the moving average.
            self.difference=cv2.absdiff( self.color_image, self.running_average_in_display_color_depth)
                        
            # Convert the image to greyscale.
            self.grey_image=cv2.cvtColor( self.difference,cv2.COLOR_BGR2GRAY)

            # Threshold the image to a black and white motion mask:
            ret,self.grey_image = cv2.threshold(self.grey_image, self.threshT, 255, cv2.THRESH_BINARY )
        
            return(self.grey_image)
                
        ##Mixture of Gaussians
        if self.subMethod =="MOG":
            self.grey_image = self.fgbg.apply(camera_imageROI,learningRate=learningRate)
            
        #Erode to remove noise, dilate the areas to merge bounded objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
        self.grey_image= cv2.morphologyEx(self.grey_image, cv2.MORPH_OPEN, kernel)
        return(self.grey_image)
    

def is_small(grey_image,width,height,minSIZE,score,oarea,noMotion=False):
    top = 0
    bottom = 1
    left = 0
    right = 1
    points = []   # Was using this to hold camera_imageROIeither pixel coords or polygon coords.
    bounding_box_list = []
    
    # Now calculate movements using the white pixels as "motion" data
    contours,hierarchy = cv2.findContours(grey_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    
    if len(contours) == 0:
            #self.noMotion flag
            noMotion=True
            return noMotion
    
    for cnt in contours:
            bounding_rect = cv2.boundingRect( cnt )
            point1 = ( bounding_rect[0], bounding_rect[1] )
            point2 = ( bounding_rect[0] + bounding_rect[2], bounding_rect[1] + bounding_rect[3] )
            bounding_box_list.append( ( point1, point2 ) )
            
    # Find the average size of the bbox (targets), then
    # remove any tiny bboxes (which are probably just noise).
    # "Tiny" is defined as any box with 1/10th the area of the average box.
    # This reduces false positives on tiny "sparkles" noise.
    box_areas = []
    for box in bounding_box_list:
            box_width = box[right][0] - box[left][0]
            box_height = box[bottom][0] - box[top][0]
            box_areas.append( box_width * box_height )
            
    average_box_area = 0.0
    if len(box_areas): average_box_area = float( sum(box_areas) ) / len(box_areas)
    
    trimmed_box_list = []
    for box in bounding_box_list:
            box_width = box[right][0] - box[left][0]
            box_height = box[bottom][0] - box[top][0]
            
            # Only keep the box if it's not a tiny noise box:
            if (box_width * box_height) > average_box_area*.3: 
                    trimmed_box_list.append( box )
                        
    #shapely does a much faster job of polygon union
    #format into shapely bounding feature
    shape_list=[]
                        
    ## Centroids of each target and hold on to target blobs
    bound_center=[]
    
    for out in trimmed_box_list:
            
            #shapely needs to boxes as minx, miny, max x maxy
            minx=out[0][0]
            miny=out[1][1]
            maxx=out[1][0]
            maxy=out[0][1]
            
            #make into a tuple
            sh_out=sg.box(minx,miny,maxx,maxy)
            shape_list.append(sh_out)
            
    #Merge boxes that touch
    casc=cascaded_union(shape_list).buffer(1)
        
    #Make an object to get the average box size
    sumbox = []
    
    if casc.type=="MultiPolygon":
            #draw shapely bounds
                for p in range(1,len(casc.geoms)):
                    b=casc.geoms[p].bounds
                    
                    #Numpy origin is top left
                    #Shapely origin is bottom left 
                    minx,miny,maxx,maxy=b
                    
                    topleft=(int(minx),int(maxy))
                    bottomright=(int(maxx),int(miny))
                    
                    #Append to summary
                    sumbox.append(casc.geoms[p].area)
                    if casc.geoms[p].area > ((width * height) * minSIZE):
                                    #Return the centroid to list, rounded two decimals
                                    x=round(casc.geoms[p].centroid.coords.xy[0][0],2)
                                    y=round(casc.geoms[p].centroid.coords.xy[1][0],2)
                                    bound_center.append((x,y))
    else:
            b=casc.bounds
            #get size 
            sumbox.append(casc.area)
            
            #to draw polygon
            minx,miny,maxx,maxy=b
            
            topleft=(int(minx),int(maxy))
            bottomright=(int(maxx),int(miny))

            #If bounding polygon is larger than the minsize, draw a rectangle
            if casc.area > ((width * height) * minSIZE):
                            x=round(casc.centroid.coords.xy[0][0],2)
                            y=round(casc.centroid.coords.xy[1][0],2)
                            bound_center.append((x,y))
    
    if len(bound_center) == 0:
        if score > 0.985 and oarea > 225:
            noMotion=False
            return noMotion
        else:
            noMotion=True
            return noMotion
    else:
        noMotion=False
        return noMotion


def get_roi(image,x,y,w,h,width,height):
    mask = np.ones(image.shape, np.bool)
#    mask[(y*height).astype(int):((y+h)*height).astype(int), (x*width).astype(int):((x+w)*width).astype(int)] = False
    mask[int(y*height):int((y+h)*height), int(x*width):int((x+w)*width)] = False
    image[mask]=0
    return image

def get_is_fauna(dft,media,dfv):
    dft1 = dft.copy()
    dfv1 = dfv.copy()
    is_fauna_list = []
    for i in range(len(dft1['frame_array_resized'])):
        x = dft1.iloc[i].xc
        y = dft1.iloc[i].yc
        w = dft1.iloc[i].wc
        h = dft1.iloc[i].hc
        dft1['roi'] = dft1.apply(lambda r: get_roi(r.frame_array_resized,x,y,w,h,
                                             r.frame_array_resized.shape[1],r.frame_array_resized.shape[0]), axis=1)
        dfv1['roi'] = dfv1.apply(lambda r: get_roi(r.frame_array_resized,x,y,w,h,
                                             r.frame_array_resized.shape[1],r.frame_array_resized.shape[0]), axis=1)
        width0 = dft1.iloc[i].frame_array_resized.shape[1]
        height0 = dft1.iloc[i].frame_array_resized.shape[0]
        first_shape = dft1['roi'].iloc[i].shape
#        width = first_shape[1]
#        height = first_shape[0]
        
#        minSIZE = float((height0/MINSIZE)*(width0/MINSIZE))/float(height*width) # tamaño minimo de especie de interes
        minSIZE = 1/(MINSIZE*MINSIZE)
        noMotion=False
        motion = []  
    
        if media == 'image':
            frames = dfv1['roi'].to_list()
        if media == 'video':
            framesdf = dfv1[dfv1['id'] == dft1.iloc[i]['id']][['item_file','roi','x','y','w','h','score']]
            framesdf['x'] = framesdf.apply(lambda r: 0 if pd.isnull(r.x) else r.x, axis=1)
            framesdf['y'] = framesdf.apply(lambda r: 0 if pd.isnull(r.y) else r.y, axis=1)
            framesdf['w'] = framesdf.apply(lambda r: 0.02 if pd.isnull(r.w) else r.w, axis=1)
            framesdf['h'] = framesdf.apply(lambda r: 0.02 if pd.isnull(r.h) else r.h, axis=1)
            frames = framesdf['roi'].to_list()
        if SUBMETHOD == 'MOG':
            if media == 'image':
                bgs = Background(SUBMETHOD,frames[0],ACCAVG,THRESHT,MOGVAR,'image')
            if media == 'video':
                bgs = Background(SUBMETHOD,frames[0],ACCAVG,THRESHT,MOGVAR)
            if len(frames) > 130:
                for image in frames[40:130]:
                    _=bgs.BackGroundSub(image,MOGLEARNING)
            else:
                for image in frames[40:]:
                    _=bgs.BackGroundSub(image,MOGLEARNING)

        elif SUBMETHOD == 'Acc':
            first_image = np.median(np.stack(frames, axis=0),axis=0).astype(int)
            bgs = Background(SUBMETHOD,first_image,ACCAVG,THRESHT,MOGVAR)
        else:
            raise ValueError('Submethod is invalid!')
            
        if media == 'image':
            grey_image = bgs.BackGroundSub(dft1.iloc[i]['roi'],MOGLEARNING)
            noMotion = is_small(grey_image,width0,height0,minSIZE,
                                dft1.iloc[i]['score'],sg.box(dft1.iloc[i].x*100,
                                                             dft1.iloc[i].y*100,
                                                             (dft1.iloc[i].x+dft1.iloc[i].w)*100,
                                                             (dft1.iloc[i].y+dft1.iloc[i].h)*100).area,noMotion)
            is_fauna_list.append(not noMotion)
        elif media == 'video':
            framesdf['is_fauna'] = framesdf.apply(lambda r: not is_small(bgs.BackGroundSub(r.roi,MOGLEARNING),
                                                                         width0,height0,minSIZE,r.score,
                                                                         sg.box(r.x*100,r.y*100,
                                                                                (r.x+r.w)*100,
                                                                                (r.y+r.h)*100).area,noMotion), axis=1)
            is_fauna_list.append(framesdf)
    if media == 'image':
        dft1['is_fauna'] = is_fauna_list
        return dft1.drop(['roi','xc','yc','wc','hc'],axis=1)
    elif media == 'video':
        dftm = pd.concat(is_fauna_list).drop(['x','y','w','h','score'], axis=1)
        dff = dfv.copy()
        dff = dff.merge(dftm, how='left', on='item_file')
        return dff.drop(['roi'], axis=1)

def run_motionmeerkat(df):
    if type(df['frame_array']) == pd.core.series.Series:
        nFrames = len(df['frame_array'])
        nvFrames = len(df[df['item_type'] == VIDEO_TYPE])
        
        id_list = df['id'].drop_duplicates().to_list()
        media_xs = [df[df['id'] == ind]['frame_array'].iloc[0][0].shape[1] for ind in id_list]
        media_ys = [df[df['id'] == ind]['frame_array'].iloc[0][0].shape[0] for ind in id_list]
        min_x = np.argmin(media_xs)
        min_y = np.argmin(media_ys)
        if min_x != min_y:
            if media_xs[min_x] < media_ys[min_y]:
                min_ind = id_list[min_x]
            else:
                min_ind = id_list[min_y]
        else:
            min_ind = id_list[min_x]
        dim = df[df['id'] == min_ind]['frame_array'].iloc[0][0].shape[1::-1]
        frames = [cv2.resize(df['frame_array'].iloc[ix][0], dim, interpolation = cv2.INTER_AREA) 
                    if df['id'].iloc[ix] != min_ind else
                  df['frame_array'].iloc[ix][0]
                  for ix in range(nFrames)]
        
        df['frame_array_resized'] = frames
        frames = []
        
        dfi = df[df['item_type'] == IMAGE_TYPE]
        dfi = dfi[dfi['detected'] == True]
        dfv = df[df['item_type'] == VIDEO_TYPE]
        dfvid = dfv[dfv['detected'] == True]['id'].drop_duplicates().to_list()
        
        dff_list = []
        is_video = True
        if len(dfv) > 0:
            for i in range(len(dfi)):
                x = dfi.iloc[i].x
                y = dfi.iloc[i].y
                w = dfi.iloc[i].w
                h = dfi.iloc[i].h
                dft = pd.DataFrame(dfi.iloc[[i]]).copy()
                dft['xc'] = x
                dft['yc'] = y
                dft['wc'] = w
                dft['hc'] = h
                dff_list.append(get_is_fauna(dft,'image',dfv))
        else:
            df['is_fauna'] = df['detected']
            is_video = False

        for i in range(len(dfvid)):
            dft = dfv[dfv['id'] == dfvid[i]].copy()
            shape_list = []
            for i in range(len(dft['frame_array_resized'])):
                if dft.iloc[i].detected:
                    sh_out=sg.box(dft.iloc[i].x,dft.iloc[i].y,dft.iloc[i].x+dft.iloc[i].w,dft.iloc[i].y+dft.iloc[i].h)
                    shape_list.append(sh_out)
            #Merge boxes that touch
            casc=cascaded_union(shape_list).buffer(1)
            
            if casc.type == 'Polygon':
                casclen = 1
            else:
                casclen = len(casc.geoms)
                
            r_list = []
            for c in range(casclen):
                flag = False
                for i in range(len(dft['frame_array_resized'])):
                    if not flag:
                        r = dft.iloc[[i]].copy()
                        if r['detected'].values:
                            ibox = sg.box(r.x.values,r.y.values,r.x.values+r.w.values,r.y.values+r.h.values)
                            if casc.type=="MultiPolygon":
#                                for p in range(len(casc.geoms)):
                                    if casc.geoms[c].contains(ibox):
                                        b= None
                                        b=casc.geoms[c].bounds
#                                        minx,miny,maxx,maxy=b
                                        r['xc'] = b[0]
                                        r['yc'] = b[1]
                                        r['wc'] = b[2] - b[0]
                                        r['hc'] = b[3] - b[1]
                                        r_list.append(pd.DataFrame(r))
                                        flag = True
                            else:
                                if casc.contains(ibox):
                                    b=None
                                    b=casc.bounds
#                                    minx,miny,maxx,maxy=b
                                    r['xc'] = b[0]
                                    r['yc'] = b[1]
                                    r['wc'] = b[2] - b[0]
                                    r['hc'] = b[3] - b[1]
                                    r_list.append(pd.DataFrame(r))
                                    flag = True
            if len(r_list) > 0:
                dft = pd.DataFrame(pd.concat(r_list)).copy()
                dff_list.append(get_is_fauna(dft,'video',dfv))

        if len(dff_list) > 0 and is_video:
            df = pd.concat(dff_list)
        elif len(dff_list) == 0 and is_video:
            df['is_fauna'] = False
                                    
    elif type(df['frame_array']) == list:
        nFrames = 1
        df['is_fauna'] = True
        
    df = df[df['is_fauna'] == True]
    df = df[df['detected'] == True]
    df = df.drop(['frame_array','is_fauna','frame_array_resized','detected'], axis=1)

    return df