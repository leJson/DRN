import json
import numpy as np

import open3d as o3d


def json_load(filename):
    with open(filename, "r") as fr:
        vars = json.load(fr)
    # for k, v in vars.items():
    #     vars[k] = np.array(v)
    return vars

def _show_gd():
    path = '/home/ljs/workspace/eccv/FirstTrainingData/label/GroundTruth.json'
    index = '50'
    contxt = json_load(path)
    print(contxt)
    contxt = contxt['Measurements']
    print(contxt)
    print(100*'*')
    item = 'Image%s' % index
    print(contxt[item])
    print(contxt[item]['Variety'])
    print(contxt[item]['RGBImage'])
    print(contxt[item]['DebthInformation'])
    print(contxt[item]['FreshWeightShoot'])
    print(contxt[item]['DryWeightShoot'])
    print(contxt[item]['Height'])
    print(contxt[item]['Diameter'])
    print(contxt[item]['LeafArea'])
    pass


def show_gd():
    path = '/home/ljs/workspace/eccv/FirstTrainingData/label/GroundTruth.json'
    index = '50'
    per_label = list()
    contxt = json_load(path)
    print(contxt)
    contxt = contxt['Measurements']
    print(contxt)
    print(100*'*')
    item = 'Image%s' % index
    print(contxt[item])
    print(contxt[item]['Variety'])
    print(contxt[item]['RGBImage'])
    print(contxt[item]['DebthInformation'])
    print(contxt[item]['FreshWeightShoot'])
    print(contxt[item]['DryWeightShoot'])
    print(contxt[item]['Height'])
    print(contxt[item]['Diameter'])
    print(contxt[item]['LeafArea'])
    per_label.append(contxt[item]['FreshWeightShoot'])
    per_label.append(contxt[item]['DryWeightShoot'])
    per_label.append(contxt[item]['Height'])
    per_label.append(contxt[item]['Diameter'])
    per_label.append(contxt[item]['LeafArea'])
    return per_label
    # pass


def read_Label(path):

    path = '/home/ljs/workspace/eccv/FirstTrainingData/label/GroundTruth.json'
    index = '50'
    contxt = json_load(path)
    print(contxt)
    contxt = contxt['Measurements']
    return contxt
    # print(contxt)
    # print(100*'*')
    # item = 'Image%s' % index
    # print(contxt[item])
    # print(contxt[item]['Variety'])
    # print(contxt[item]['RGBImage'])
    # print(contxt[item]['DebthInformation'])
    # print(contxt[item]['FreshWeightShoot'])
    # print(contxt[item]['DryWeightShoot'])
    # print(contxt[item]['Height'])
    # print(contxt[item]['Diameter'])
    # print(contxt[item]['LeafArea'])
    # pass


def read_pcd_pointclouds(file_path):
    # file_path = '/home/ljs/workspace/eccv/FirstTrainingData/out_4096/train/38.pcd'
    pcd = o3d.io.read_point_cloud(file_path)
    point_cloud = np.asarray(pcd.points)
    # color_cloud = np.asarray(pcd.colors)*255
    color_cloud = np.asarray(pcd.colors)
    points = np.concatenate([point_cloud, color_cloud], axis=1)
    # print(point_cloud.shape)
    # print(color_cloud.shape)
    print(points.shape)
    return points
    # print(color_cloud)
    # np.savetxt('38.txt', points, fmt='%10.8f') # Keep 8 decimal places
    # pass

def mse_cal():
    a = np.asarray([[1,2,4], [2,3,5]])
    b = np.asarray([[0,0,4], [2,3,5]])
    # print(a.shape)
    print((a-b)*(a-b))

if __name__ == '__main__':
    mse_cal()
    # read_pcd_pointclouds()
    # show_gd()

    # file_path = '/home/ljs/workspace/eccv/FirstTrainingData/out_4096/train/38.pcd'
    # read_pcd_pointclouds(file_path)
