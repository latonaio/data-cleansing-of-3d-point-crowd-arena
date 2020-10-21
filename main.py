import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import aion.common_library as common
import grpc
import numpy as np
import open3d as o3d
from StatusJsonPythonModule.StatusJsonRest import StatusJsonRest

from api import PointCloud_pb2, PointCloud_pb2_grpc, pcdproto
from api.const import NEXTSERVICE

OUTPUTPATH = common.get_output_path(os.getcwd(), __file__)
INTERVAL = 1
DOWN_SCALE = 1
ERROR_COUNT_MAX = 100


def save_point_cloud(xyz, timestamp):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    filepath = os.path.join(OUTPUTPATH, f'{timestamp}.ply')
    o3d.io.write_point_cloud(filepath, pcd, write_ascii=True)
    print("save point cloud to ", filepath)
    return filepath


def identify_outliers_index(npcd):
    quartile_1, quartile_3 = np.percentile(npcd, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    # print("(lower_bound, upper_bound) = ", (lower_bound, upper_bound))
    return np.where((npcd < lower_bound) | (npcd > upper_bound))


def delete_outlier(npcd):
    INVALID_VALUE = 16383
    xyz = deepcopy(npcd)
    xyz = np.delete(xyz, np.where(npcd > INVALID_VALUE)[0], axis=0)
    print(npcd[0:2])
    for i in range(npcd.shape[1]):
        _indexes = identify_outliers_index(xyz[:, i])
        xyz = np.delete(xyz, _indexes[0], axis=0)
    return xyz


def identify_indexes_outside_percentage(npcd, one_in_x):
    target_percent = 100 / one_in_x
    lower = (100 - target_percent) / 2
    upper = 100 - lower
    lower_bound, upper_bound = np.percentile(npcd, [lower, upper])

    return np.where((npcd < lower_bound) | (npcd > upper_bound))


def scale_down(npcd, downscale=1, axis=2):
    xyz = deepcopy(npcd)

    if axis == 0 or axis == 1:
        _indexes = identify_indexes_outside_percentage(xyz[:, axis], downscale)
        xyz = np.delete(xyz, _indexes[0], axis=0)
    elif axis == 2:
        sqrt_x = np.sqrt(downscale)
        for i in range(2):
            _indexes = identify_indexes_outside_percentage(
                xyz[:, i], sqrt_x)
            xyz = np.delete(xyz, _indexes[0], axis=0)

    return xyz


class PointCloudClient():

    def __init__(self, host='127.0.0.1', port=50051):
        self.host = host
        self.port = port

    def get_point_cloud(self):
        address = f'{self.host}:{self.port}'
        points = []
        timestamp = None
        with grpc.insecure_channel(address) as channel:
            try:
                stub = PointCloud_pb2_grpc.MainServerStub(channel)
                responses = stub.get_point_cloud(
                    PointCloud_pb2.PointRequest())
                for res in responses:
                    nda = pcdproto.proto_to_ndarray(res)
                    points.append(nda)
                    timestamp = res.timestamp

            except grpc.RpcError as e:
                print(e.details())
                raise RuntimeError(e.details())

        if len(points) > 0:
            npcd = np.concatenate(points)
            return npcd, timestamp
        else:
            raise RuntimeError('get no point cloud data')
            return [], ''


def main():

    ####################
    # AION HOME
    ####################
    home_dir = Path.home()
    device_name = __file__.replace(f'{home_dir}/', '').split('/')[0]
    aion_home = os.path.join(home_dir, device_name)

    ####################
    # status json
    ####################
    status_obj = StatusJsonRest(os.getcwd(), __file__)
    status_obj.initializeInputStatusJson()
    status_obj.setNextService(
        NEXTSERVICE['name'],
        f'{aion_home}/Runtime/{NEXTSERVICE["directory"]}',
        'python', 'main.py'
    )

    ##################
    # main process
    #################
    client = PointCloudClient()
    err_count = 0
    while True:
        try:
            npcd, timestamp = client.get_point_cloud()
        except RuntimeError as e:
            print(e)
            err_count += 1
            if err_count > ERROR_COUNT_MAX:
                print("can not get 3d point cloud")
                break
            time.sleep(0.1)
            continue

        err_count = 0
        npcd = delete_outlier(npcd)
        if DOWN_SCALE > 1:
            npcd = scale_down(npcd, downscale=DOWN_SCALE, axis=1)
            # print('down scale :', DOWN_SCALE)

        filepath = save_point_cloud(npcd, timestamp)

        ##################
        # A json
        ##################
        status_obj.setMetadataValue(
            'pointcloud', {
                'filepath': filepath,
                'timestamp': timestamp
            }
        )

        status_obj.outputJsonFile()
        status_obj.resetOutputJsonFile()
        time.sleep(INTERVAL)


if __name__ == '__main__':
    main()
