import os

import pandas as pd
import spacenet_ann_extractor
import numpy as np

ROOT_DIR = os.getcwd()

CSV_PATH = os.path.join(ROOT_DIR, "summaryData")
dest_dir_name = "ground_truth_files_yolo"

if os.path.isdir(ROOT_DIR + os.sep + dest_dir_name) is False:
    os.mkdir(ROOT_DIR + os.sep + "{}".format(dest_dir_name))

DEST_PATH = os.path.join(ROOT_DIR, dest_dir_name)

im_ext = ['.jpg', '.tif', '.png']

img_path = ROOT_DIR + os.sep + "images"

img_files = [i for ind in os.walk(img_path) for i in ind[2] if i[-4:].lower() in im_ext]



# Create the ground truth file for the image
for ind, each_image in enumerate(img_files):

    print('{}/{}'.format(ind, len(img_files)))
    
    # Get rows with the specific image id
    IMG_ID = img_files[ind][:-4] # Atlanta_nadir44_catid_1030010003CCD700_745301_3733239
    BLDG_ID = IMG_ID[-14:]#re.search("_\d+_\d+",IMG_ID).group()[1:] # 745301_3733239
    CSV_ID = IMG_ID[:-15]#re.sub(BLDG_ID, '', IMG_ID)[:-1] # Atlanta_nadir44_catid_1030010003CCD700
    
    df = pd.read_csv(CSV_PATH + os.sep + CSV_ID + "_Train.csv")
    df = df.drop(["Unnamed: 0"],axis=1)
    
    df2 = df.loc[df['ImageId'] == each_image[:-4]]
    num_bldgs = len(df2)
    
    # Create the ground truth dataframe
    d = {'Class_Name': [0] * num_bldgs,
         'xcenter'   : [0] * num_bldgs,
         'ycenter'   : [0] * num_bldgs,
         'width'     : [0] * num_bldgs,
         'height'    : [0] * num_bldgs}
    
    gt = pd.DataFrame(data=d,dtype=float)
    
    for each_bldg in range(num_bldgs):
        # Get bbox coordinate of this image
        polypix = spacenet_ann_extractor.load_polypixel_coordinates(CSV_PATH, CSV_ID, BLDG_ID, each_bldg)
        bbox = spacenet_ann_extractor.get_bbox_coordinates(polypix,bbox_format="CWH")
        
        gt.at[each_bldg, 'xcenter'] = bbox[0] / 900
        gt.at[each_bldg, 'ycenter'] = bbox[1] / 900
        gt.at[each_bldg, 'width'] = bbox[2] / 900
        gt.at[each_bldg, 'height'] = bbox[3] / 900
        
    np.savetxt(DEST_PATH + os.sep + each_image[:-4] + '.txt', gt.values, fmt=["%i","%f","%f","%f","%f"])
