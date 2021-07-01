import os
import json
import csv
import math
from collections import defaultdict
import numpy as np
import cv2

class CoordMapper:
    def __init__(self, camera_config_file):
        self._camera_parameters = defaultdict(dict)
        self._camera_configs = json.load(open(os.path.join(camera_config_file)))
        self._camera_ids = list()

        for camera_param in self._camera_configs['Cameras']:

            cam_id = camera_param['CameraId']
            self._camera_ids.append(cam_id)

            self._camera_parameters[cam_id]['Translation'] = np.asarray(
                camera_param['ExtrinsicParameters']['Translation'])
            self._camera_parameters[cam_id]['Rotation'] = np.asarray(
                camera_param['ExtrinsicParameters']['Rotation']).reshape((3, 3))

            self._camera_parameters[cam_id]['FInv'] = np.asarray([
                1 / camera_param['IntrinsicParameters']['Fx'],
                1 / camera_param['IntrinsicParameters']['Fy'], 1
            ])
            self._camera_parameters[cam_id]['C'] = np.asarray([
                camera_param['IntrinsicParameters']['Cx'],
                camera_param['IntrinsicParameters']['Cy'], 0
            ])

        self._discretization_factorX = 1.0 / (
            (self._camera_configs['Space']['MaxU'] - self._camera_configs['Space']['MinU']) / (math.floor(
                (self._camera_configs['Space']['MaxU'] - self._camera_configs['Space']['MinU']) /
                self._camera_configs['Space']['VoxelSizeInMM']) - 1))
        self._discretization_factorY = 1.0 / (
            (self._camera_configs['Space']['MaxV'] - self._camera_configs['Space']['MinV']) / (math.floor(
                (self._camera_configs['Space']['MaxV'] - self._camera_configs['Space']['MinV']) /
                self._camera_configs['Space']['VoxelSizeInMM']) - 1))
        self._discretization_factor = np.asarray([self._discretization_factorX, self._discretization_factorY, 1])

        self._min_volume = np.asarray([
            self._camera_configs['Space']['MinU'], self._camera_configs['Space']['MinV'],
            self._camera_configs['Space']['MinW']
        ])

    def projection(self, person_center, camera_id):
        '''
        project person footpoint in topdown view back to camera coordinate.

        Arguments:
            camera_id: str, id of camera.
            person_center: X and Y coordinates of person in topdown view.
        '''

        topdown_coords = np.transpose(
            np.asarray([[person_center['X'], person_center['Y'], 0]]))
        world_coord = topdown_coords / self._discretization_factor[:, np.newaxis] + self._min_volume[:, np.newaxis]
        uvw = np.linalg.inv(self._camera_parameters[camera_id]['Rotation']) @ (
            world_coord - self._camera_parameters[camera_id]['Translation'][:, np.newaxis])
        pixel_coords = uvw / self._camera_parameters[camera_id]['FInv'][:, np.newaxis] / uvw[
            2, :] + self._camera_parameters[camera_id]['C'][:, np.newaxis]
        return pixel_coords[0][0], pixel_coords[1][0]


class DataReader:
    def __init__(self, image_dir, label_dir, num_cam) -> None:
        self.label_dir = label_dir
        self._image_dir = image_dir
        self._num_cam = num_cam
        self._mapper = CoordMapper('/mnt/sdb/mmp_public/calibrations/retail/calibrations.json')
    
    def read(self, frame_id):
        tracklets = []
        with open(os.path.join(self.label_dir, 'topdown_'+str(frame_id).zfill(5)+'.csv'), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                tracklets.append({'center_coordinate':np.asarray([int(float(row[2])), int(float(row[1]))]), 'id':int(row[0])})
        rgbs = {}
        for cam_id in range(1, self._num_cam+1):
            rgbs[cam_id] = cv2.imread(os.path.join(self._image_dir, 'rgb_'+str(frame_id).zfill(5)+'_'+str(cam_id)+'.jpg'), cv2.IMREAD_UNCHANGED)
        
        self.plot(tracklets, rgbs)
        return rgbs, tracklets
    
    def plot(self, tracklets, rgbs):
        for cam_id in range(1, self._num_cam+1):
            rgb = rgbs[cam_id]
            for tracklet in tracklets:
                x, y = self._mapper.projection({'X': tracklet['center_coordinate'][0], 'Y': tracklet['center_coordinate'][1]}, cam_id)
                rgb = cv2.circle(rgb, (int(x), int(y)), 5, (0, 0, 255), 2)
            cv2.imwrite(os.path.join('output', 'camera_view_'+str(cam_id)+'.jpg'), rgb)

if __name__ == '__main__':
    image_dir = '/mnt/sdb/mmp_public/images/63am/retail_0'
    topdown_label_dir = '/mnt/sdb/mmp_public/topdown_labels/63am/retail_0'
    num_cam = 6
    data_reader = DataReader(image_dir, topdown_label_dir, num_cam)
    data_reader.read(100)