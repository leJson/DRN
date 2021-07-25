# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 23:20:13 2020
@author: li jing song
This file contain some functions for picture processing
"""
import shutil
import sys
import os
import cv2
import numpy as np
import json
# import pyrealsense2 as rs
import time
import random
from matplotlib import pyplot as plt
import glob
def check_path(file_path):
    """
    Check whether the file is existing
    :param file_path:
    :return:
    """
    if not (os.path.exists(file_path)):
        print('file is not existence')
        sys.exit()


def show_image(img, time=5000):
    """
    Show a img mat in normal way.
    """
    cv2.namedWindow('Licence Img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Licence Img', 1280, 768)
    cv2.moveWindow('Licence Img', 300, 100)
    cv2.imshow('Licence Img', img)
    if time > 0:
        cv2.waitKey(time)
        cv2.destroyAllWindows()


def rgb_to_gray(img):
    """
    Convert bgr image to q grey one, and return them.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binary_thresh_image(gray_img, thresh1=0, thresh2=255):
    """
    Get binary img from a gray img
    :param thresh1: low thresh
    :param thresh2: high thresh
    :param gray_img: gray mat
    :return: thresh value; binary mat
    """
    ret, binary_img = cv2.threshold(gray_img, thresh1, thresh2, cv2.THRESH_BINARY)
    return binary_img


def auto_binary_thresh_image(gray_img):
    """
    Get binary img from a gray img in mean index value thresh
    :param gray_img: gray mat
    :return: thresh value; binary mat
    """
    max_index = float(gray_img.max())
    min_index = float(gray_img.min())
    thresh1 = (max_index + min_index) / 2
    thresh2 = 255
    ret, binary_img = cv2.threshold(gray_img, thresh1, thresh2, cv2.THRESH_BINARY)
    return binary_img


def stretch(img):
    """
    图像拉伸函数
    """
    maxi = float(img.max())
    mini = float(img.min())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = (255 / (maxi - mini) * img[i, j] - (255 * mini) / (maxi - mini))

    return img


def horizon_stack_image(*args):
    """
    stack array
    :param args:
    :return:
    """
    img_mix = args[0]
    for i in range(1, len(args)):
        img_mix = np.hstack((img_mix, args[i]))
    return img_mix


def vertical_stack_image(*args):
    """
    stack array
    :param args:
    :return:
    """
    img_mix = args[0]
    for i in range(1, len(args)):
        img_mix = np.vstack((img_mix, args[i]))
    return img_mix


def erode_image(img, kernel, iterations=1):
    """
    Erode image
    :param img:
    :param kernel:
    :param iterations:
    :return:
    """
    return cv2.erode(img, kernel, iterations)


def dilate_image(img, kernel, iterations=1):
    """
    Dilate image
    :param img: mat
    :param kernel:
    :param iterations:
    :return: a dilated images
    """
    return cv2.dilate(img, kernel, iterations)


def abs_diff(img1, img2):
    """
    get a abs diff image from two images
    :param img1: mat1
    :param img2: mat2
    :return: diff results
    """
    return cv2.absdiff(img1, img2)


def canny_detect(img, thresh1, thresh2):
    """
    Cannon detects
    :param img:
    :param thresh1:
    :param thresh2:
    :return:
    """
    return cv2.Canny(img, thresh1, thresh2)


def divide_LeftRight(img):
    """
    Divide bi-camera image into left image and right image
    bi-camera image shape:(720,2560,3); left and right image shape:(720,1280,3)
    :param img: CV2 mat image
    :return: ;left_img, right_img
    """
    divide_index = img.shape[1] // 2
    right_img = img[:, 0:divide_index]
    left_img = img[:, divide_index:]
    return left_img, right_img


def ruihua(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 定义一个核
    dst = cv2.filter2D(img, -1, kernel=kernel)
    return dst


def adaptive_equalizeHist(img, index=3):
    'Adaptive Histogram Equalization'
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(index, index))

    if len(img.shape) == 2:
        return clahe.apply(img)
    else:
        (b, g, r) = cv2.split(img)
        bH = clahe.apply(b)
        gH = clahe.apply(g)
        rH = clahe.apply(r)
        return cv2.merge((bH, gH, rH))


def medianBlur(img,index=3):
    '中值滤波'
    return cv2.medianBlur(img, index)


def img_retify(img,mtx,dist):

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return dst


def _json_load(filename):
    with open(filename, "r") as fr:
        vars = json.load(fr)
    for k, v in vars.items():
        vars[k] = np.array(v)
    return vars


def json_load(filename):
    with open(filename, "r") as fr:
        vars = json.load(fr)
    # for k, v in vars.items():
    #     vars[k] = np.array(v)
    return vars


def write_txt(fname='123.txt', contxt='str'):
    """
    :param fname:
    :param contxt:
    :return:
    """
    if os.path.exists(fname):
        with open(fname, "a") as file:
            file.write(contxt)
    else:
        with open(fname, "w") as file:
            file.write(contxt)
    return 0


def point_img(img, x, y, thick=-1):
    """
    :param img:
    :param x: x location
    :param y: y location
    :param thick:
    :return:
    """
    cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=thick)
    print('img.shape', img.shape)
    print('img[x][y][0],img[x][y][1],img[x][y][2]', img[y][x][0], img[y][x][1], img[y][x][2])
    # img[y][x][0] = 0
    # img[y][x][1] = 0
    # img[y][x][2] = 255
    return img


def get_feature_points(img):
    """
    get feature points by Shi-Tomasi
    :param img: img mat
    :return: feature points
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature_params = dict(maxCorners=8000000,
                          qualityLevel=0.002,
                          minDistance=0,
                          blockSize=5,
                          useHarrisDetector=True,
                          )

    #img = cv2.bilateralFilter(img, 16, 50, 50)
    points = cv2.goodFeaturesToTrack(img, mask=None, **feature_params)
    return points
'''
def get_surf_feature_points(img1, img2):
    """
    get feature points by surf
    :param img1:
    :param img2:
    :return: surf feature points
    """
    surf = cv2.xfeatures2d.SURF_create(20)
    key_points = surf.detect(img1, None)
    return key_points
'''

def lk_optical_flow(img1, img2, points):
    """
    get match points from 2 img by lucas optical flow
    :param img1: img1
    :param img2: img2
    :param points: key points
    :return: matched key points
    """
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    new_points, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, points, None, **lk_params)
    return new_points


def un_distort_img(distort_img, mtx, dist):

    """
    use camera inner parameters to un-distort image
    :param distort_img:
    :param mtx:
    :param dist:
    :return: normal image
    """
    img = cv2.undistort(distort_img, mtx, dist, None, mtx)
    return img


def findEssentialMat(points1, points2, camMatrix):
    """
    :param points1: left_image key points
    :param points2: right_image key points
    :param camMatrix: camMatrix
    :return: E
    """
    e, mask = cv2.findEssentialMat(points1, points2, camMatrix, method=cv2.RANSAC, prob=0.999, threshold=3.0)
    return e


def lsr_rectifyImagePair(imgL, imgR, cam_mtx, cam_dist, r, t):
    """
    :param imgL: cv2.image(np.array)
    :param imgR: cv2.image(np.array)
    :param imageShape: py.tuple, (h, w)
    :param CameraParams: py.dict, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T
    :return: rectifiedImgL, rectifiedImgR: cv2.image(np.array)
    """
    assert imgL.shape == imgR.shape, "imgL.shape != imgR.shape."
    h, w = imgL.shape[:2]
    print('w,h:',w,h)
    cameraMatrix1, distCoeffs1 = cam_mtx, cam_dist
    cameraMatrix2, distCoeffs2 = cam_mtx, cam_dist
    R, T = r, t
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
        cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)
    rectifiedImgL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectifiedImgR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)
    print(' rectifiedImgR:', rectifiedImgR.shape)
    return rectifiedImgL, rectifiedImgR


def get_surf_feature_points(im1,im2):
    """
    :param im1:
    :param im2:
    :return:
    """
    surf=cv2.xfeatures2d.SURF_create()
    kp1,des1=surf.detectAndCompute(im1,None)
    kp2,des2=surf.detectAndCompute(im2,None)
    FLANN_INDEX_KDTREE = 0
    index_p=dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    searth_p=dict(checks=500)
    flann=cv2.FlannBasedMatcher(index_p,searth_p)
    matches=flann.knnMatch(des1,des2,k=2)
    good =[]
    pts1=[]
    pts2=[]
    for i,(m,n) in enumerate(matches):
        if m.distance  < 5*n.distance: #defualt 0.6
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    pts1=np.float32(pts1)
    pts2= np.float32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 0.5) #dedualt 0.01
    points1 = pts1[mask.ravel() == 1]
    points2 = pts2[mask.ravel() == 1]
    return points1, points2


def open_morphology(img, size=(3, 3)):
    """
    :param img:
    :param size:
    :return:
    """
    kernel = np.ones(size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def close_morphology(img, size=(3, 3)):
    """
    :param img:
    :param size:
    :return:
    """
    kernel = np.ones(size, np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def find_corner(img, board_size=(9, 6)):
    """
    :param img:
    :param board_size:
    :return:
    """
    ret, corners = cv2.findChessboardCorners(img, board_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    return ret, corners


def find_cornerSubPix_withcorners(img, corners, windsize=(11, 11)):
    """
    :param img:
    :param corners:
    :param windsize:
    :return:
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    subpix_corners = cv2.cornerSubPix(img, corners, windsize, (-1, -1), criteria)
    return subpix_corners


def find_cornerSubPix(img, board_size=(9, 6), windsize=(11, 11)):
    """
    :param img:
    :param board_size:
    :param windsize:
    :return:
    """
    ret, corners = cv2.findChessboardCorners(img, board_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        subpix_corners = cv2.cornerSubPix(img, corners, windsize, (-1, -1), criteria)
    else:
        print('no corner was founded!')
    return subpix_corners


def draw_corners(img, corners):
    """
    :param img:
    :param corners:
    :return:
    """
    for corner in corners:
        pointed_img = point_img(img, round(corner[0][0]), round(corner[0][1]), thick=1)
    return pointed_img


def div_manual_corners(array):
    """
    :param array:
    :return:
    """
    h, w = array.shape
    h_array = np.zeros([h//2, w])
    w_array = np.zeros([h//2, w])
    for i in range(h):
        if i % 2 == 0:
            w_array[i // 2, :] = array[i, :]
        if i % 2 == 1:
            h_array[i // 2, :] = array[i, :]
    return h_array, w_array


def chess_board_corners(corners, width=9):
    '''
    :param corners:
    :param width:
    :return:
    '''
    c_len = corners.shape[0]
    de_h_array = np.zeros([c_len//width, width])
    de_w_array = np.zeros([c_len//width, width])
    for i in range(c_len):
        h = i//9
        w = i % 9
        de_w_array[h][w] = round(corners[c_len - i - 1][0][0])
        de_h_array[h][w] = round(corners[c_len - i - 1][0][1])
    return de_h_array, de_w_array


def calculate_points_cloud(depth_frame):
    """
    useless
    :param depth_frame:
    :return:
    """
    pc = rs.pointcloud()
    pc.as_spatial_filter()
    time1 = time.time()
    points = pc.calculate(depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    np.savetxt('vtx.txt', vtx)
    time2 = time.time()
    print('cost_time:', time2 - time1)
    print('vtx.shape:', vtx.shape)
    return vtx


def spatial_filter(depth_frame):
    """
    useless
    :param depth_frame:
    :return:
    """
    pc = rs.pointcloud()
    filtered = pc.as_spatial_filter(depth_frame)
    return filtered


def get_real_distance(point1, point2, point_cloud_map):
    '''
    :param point1:
    :param point2:
    :param point_cloud_map:
    :return:
    '''

    h1, w1 = point1
    h2, w2 = point2

    x1, y1, z1 = point_cloud_map[0][h1][w1], point_cloud_map[1][h1][w1], point_cloud_map[2][h1][w1]
    x2, y2, z2 = point_cloud_map[0][h2][w2], point_cloud_map[1][h2][w2], point_cloud_map[2][h2][w2]
    print('x1, y1, z1 ',x1, y1, z1)
    print('x2, y2, z2 ', x2, y2, z2)
    distance = pow(pow(x1-x2, 2)+pow(y1-y2, 2)+pow(z1-z2, 2), 0.5)
    return distance


def chess_board_distance(point1, point2, h_scala=26.58, w_scala=26.67):
    '''
    :param point1:
    :param point2:
    :param h_scala: 每个格子在h方向的大小，单位mm
    :param w_scala: 每个格子在w方向的大小，单位mm
    :return:
    '''
    #h_scala = 26.58
    #w_scala = 26.67
    h1, w1 = point1
    h2, w2 = point2
    distance = pow(pow((h1 - h2)*h_scala, 2) + pow((w1 - w2)*w_scala, 2), 0.5)
    return distance


def get_random_hw(max_h=6, max_w=9):
    '''
    :param max_h:
    :param max_w:
    :return:
    '''
    rand0m_h = random.randint(0, max_h-1)
    rand0m_w = random.randint(0, max_w - 1)
    return [rand0m_h, rand0m_w]


def get_random_hw_pair_forclass(num=10, hw_array=np.array([]), choosed=True):
    '''
    :param choosed:
    :param num:
    :param hw_array:
    :return:
    '''

    max_h = 6
    max_w = 9
    _num = 0
    hw_pairs = list()
    h0, w0 = get_random_hw(max_h, max_w)
    hw_pairs.append([h0, w0])
    while _num < num-1:
        front_h, front_w = hw_pairs[-1]
        temp_h, temp_w = get_random_hw(max_h, max_w)
        if choosed:
            judge = hw_array[2][temp_h][temp_w] < 3
        else:
            judge = True
        if abs(front_h-temp_h) + abs(front_w - temp_w) > 0 and judge:
            hw_pairs.append([temp_h, temp_w])
            _num += 1
        else:
            print('temp_h:%s, temp_w:%s is invalid !' % (temp_h, temp_w), 100*'#')

    return np.asanyarray(hw_pairs)


def get_random_hw_pair(num=10, hw_array=np.array([]),choosed=True):
    '''
    :param choosed:
    :param num:
    :param hw_array:
    :return:
    '''

    max_h = 6
    max_w = 9
    _num = 0
    hw_pairs = list()
    h0, w0 = get_random_hw(max_h, max_w)
    hw_pairs.append([h0, w0])
    while _num < num-1:
        front_h, front_w = hw_pairs[-1]
        temp_h, temp_w = get_random_hw(max_h, max_w)
        if abs(front_h-temp_h) + abs(front_w - temp_w) > 0 :
            hw_pairs.append([temp_h, temp_w])
            _num += 1
        else:
            print('temp_h:%s, temp_w:%s is invalid !' % (temp_h, temp_w), 100*'#')

    return np.asanyarray(hw_pairs)


def distance_check(hw_array, point_cloud_map, test_num=100, h_level=2):
    diff_list = list()
    for i in range(test_num):
        random_seed_list = get_random_hw_pair(num=h_level, hw_array=hw_array, choosed=True)
        point1 = random_seed_list[0]
        point2 = random_seed_list[1]
        #print('point1,point2:', point1, point2)
        rgb_point1 = [int(hw_array[0][point1[0]][point1[1]]), int(hw_array[1][point1[0]][point1[1]])]
        rgb_point2 = [int(hw_array[0][point2[0]][point2[1]]), int(hw_array[1][point2[0]][point2[1]])]
        #print('rgb_point1,rgb_point2:', rgb_point1, rgb_point2)
        real_distance = get_real_distance(rgb_point1, rgb_point2, point_cloud_map)*1000
        #print('real_distance', real_distance)

        chessboard_distance = chess_board_distance(point1, point2)
        #print('chessboard_distance', chessboard_distance)
        differ = abs(real_distance - chessboard_distance)
        #print('real_distance, chessboard_distance', real_distance, chessboard_distance)
        diff_list.append(differ)

        print(differ)
    return diff_list


def average_calculus_diff(diff_res):
    '''
    :param diff_res:
    :return:
    '''
    diff_res = np.array(diff_res)
    len_list = len(diff_res)
    res = list()
    for i in range(len_list):
        aver = diff_res[:i+1].sum()/(i+1)
        res.append(aver)
    return res


def glob_image_dir(path='save_img1', cap='save'):
    """
    :param path:
    :param cap:
    :return:
    """
    img_paths = glob.glob(path + '/%s*' % cap)
    return img_paths


def draw_corners(img, corners):
    """
    :param img:
    :param corners:
    :return:
    """
    for corner in corners:
        pointed_img = point_img(img, round(corner[0][0]), round(corner[0][1]), thick=1)
    return pointed_img


def plot_show(num=2, *args):
    '''
    for example :  plot_show(3, cuted_img, cuted_img, cuted_img, cuted_img, cuted_img)
    :param num: the number of image in each raw
    :param args:
    :return:
    '''
    raw_num = len(args)//num
    index = 0
    if len(args) % num > 0:
        index = 1
    for i in range(len(args)):
        plt.subplot(raw_num+index, num, i+1)
        plt.imshow(args[i], cmap=plt.cm.gray)
        plt.axis('off')
    plt.show()


def plot_show_hot(num=2, *args):
    '''
    for example :  plot_show(3, cuted_img, cuted_img, cuted_img, cuted_img, cuted_img)
    :param num: the number of image in each raw
    :param args:
    :return:
    '''
    raw_num = len(args)//num
    index = 0
    if len(args) % num > 0:
        index = 1
    for i in range(len(args)):
        plt.subplot(raw_num+index, num, i+1)
        plt.imshow(args[i], cmap=plt.cm.jet)
        plt.axis('off')
    plt.show()


def glob_image_dir(path='',  cap=''):

    """
    :param path:
    :param cap:
    :return:
    """

    img_paths = glob.glob(path+'/%s*' % cap)
    return img_paths


def get_outline(mask):
    """
    :param mask:
    :return:
    """
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel_8 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilate = cv2.dilate(mask, kernel=kernel_8)
    erode = cv2.erode(mask, kernel=kernel_5)
    outline = dilate - erode
    return outline


def mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)


def mix_img(img1, img2, alpha=0.5, gamma=0):
    beta = 1 - alpha
    mix = cv2.addWeighted(img1, alpha, img2, beta, gamma)
    return mix


if __name__ == '__main__':
    list_test = [1, 2, 3, 45, 33, 3]
    plt.plot(list_test)
    plt.show()

    #average_calculus_diff(list_test)
