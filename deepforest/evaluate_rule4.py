import cv2
import numpy as np
import pandas as pd
from os.path import join
from glob import glob
from os.path import join
from os.path import basename

path = "images"

# green
grn_lower_color_limit = np.array([18, 47, 0])
grn_upper_color_limit = np.array([44, 161, 227])

# brown
brn_lower_color_limit = np.array([2, 68, 224])
brn_upper_color_limit = np.array([20, 132, 255])


def calc_per(test_df, raster, fname, color):
    perc_list = []
    df = test_df[test_df.image_path == fname]
    for idx, row in df.iterrows():
        bbox = raster[row['ymin']:row['ymax'], row['xmin']:row['xmax'], 2]
        bbox_green_count = np.count_nonzero(bbox)
        perc_list.append(bbox_green_count/row['area'])
    df[color] = perc_list
    return df

def calc_bbox_area(df):
    df['area'] = (df.xmax - df.xmin) * (df.ymax - df.ymin)
    return df

def draw_rect(raster, df, color):
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for idx, row in df.iterrows():
        cv2.rectangle(raster, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 165, 255), 1)
        cv2.putText(raster, "{:.2f}".format(row[color]), (np.int((row['xmin'] + row['xmax'])/2) - 13, row['ymin'] + 10), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        #cv2.putText(raster, "{:.2f}".format(row[-1]), (np.int((row[1] + row[3]) / 2) - 12, np.int((row[2] + row[4]) / 2) + 6), font, 0.6, (0, 0, 0), 1,cv2.LINE_AA)
    return raster

def single_img_stats(path, ann_fname, img_fname, lower_color_limit, upper_color_limit, color):
    # gets percentage of pixels in a bbox with a certain color for a single image
    img_bgr = cv2.imread(join(path, img_fname))
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    test_df = pd.read_csv(ann_fname)

    mask = cv2.inRange(img_hsv, lower_color_limit, upper_color_limit)
    res = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    # cv2.imshow('image', img_bgr)
    # cv2.imshow('mask',mask)
    # cv2.imshow('res',cv2.cvtColor(res, cv2.COLOR_HSV2BGR))
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()
    test_df = calc_bbox_area(test_df)
    test_df_reduced = calc_per(test_df, res, img_fname, color)

    final_img = draw_rect(img_bgr, test_df_reduced, color)

    cv2.imshow('image', final_img)
    cv2.waitKey(0)

    #cv2.imwrite("TEAK_062_2018_p_b_green_labeled.png", final_img)

def multi_img_stats(path, ann_df, lower_color_limit, upper_color_limit, color):
    # gets stats for several images in a dataframe
    temp_df_list = []
    ann_df = calc_bbox_area(ann_df)

    ann_df['xmin'] = np.int32(ann_df['xmin'])
    ann_df['xmax'] = np.int32(ann_df['xmax'])
    ann_df['ymin'] = np.int32(ann_df['ymin'])
    ann_df['ymax'] = np.int32(ann_df['ymax'])

    # ann_df convert bbox coords to integers

    img_files = glob(join(path, "TEAK_0??_2018.tif"))

    for img_fname in img_files:
        img_bgr = cv2.imread(img_fname, cv2.IMREAD_UNCHANGED)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, lower_color_limit, upper_color_limit)
        res = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        temp_df = calc_per(ann_df, res, basename(img_fname), color)
        temp_df_list.append(temp_df)

    final_df = pd.concat(temp_df_list)
    return final_df

def count_stuff(df1, df2):
    color_lim = 0.20
    num_green = 0
    num_brown = 0
    num_both = 0
    tot_num_trees = 0
    num_green_matched = 0
    num_brown_matched = 0
    num_both_matched = 0

    tot_num_trees = df1.shape[0]
    num_green = df1[df1.green >= 0.2].shape[0]
    num_brown = df1[df1.brown >= 0.2].shape[0]
    num_both = df1[(df1.green >= 0.2) | (df1.brown >= 0.2)].shape[0]

    num_green_matched = df2[df2.green >= 0.2].shape[0]
    num_brown_matched = df2[df2.brown >= 0.2].shape[0]
    num_both_matched = df2[(df2.green >= 0.2) | (df2.brown >= 0.2)].shape[0]
    res_dict = {"num_green": num_green, "num_brown": num_brown,
                "num_both": num_both, "tot_num_trees": tot_num_trees,
                "num_green_matched": num_green_matched,
                "num_brown_matched": num_brown_matched,
                "num_both_matched": num_both_matched}
    return res_dict

def stats(test_df, matches_df):
    res_dict = count_stuff(test_df, matches_df)

    return res_dict
