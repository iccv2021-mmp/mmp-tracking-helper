import os
import json
import cv2

class DataReader:

    def __init__(self, image_dir, label_dir, num_cam, start_frame=0, end_frame=0, save_videos=False):
        self._image_dir = image_dir
        self._label_dir = label_dir
        self._num_cam = num_cam
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._save_videos = save_videos

        if self._save_videos:
            self._rgb_videos = {}
            for cam_id in range(1, self._num_cam+1):
                self._rgb_videos[cam_id] = cv2.VideoWriter('video_camera_view_'+str(cam_id)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (640, 360))
    
    def __del__(self):
        if self._save_videos:
            for _, out_video in self._rgb_videos.items():
                out_video.release()

    def read(self, frame_id):
        rgbs = {}
        labels = {}
        for cam_id in range(1, self._num_cam+1):
            rgb = cv2.imread(os.path.join(self._image_dir, 'rgb_'+str(frame_id).zfill(5)+'_'+str(cam_id)+'.jpg'), cv2.IMREAD_UNCHANGED)
            label = json.load(open(os.path.join(self._label_dir, 'rgb_'+str(frame_id).zfill(5)+'_'+str(cam_id)+'.json')))
            rgbs[cam_id] = rgb
            labels[cam_id] = label

        return rgbs, labels
    
    def plot(self):
        for frame_id in range(self._start_frame, self._end_frame):
            rgbs, labels = self.read(id)
            for cam_id in range(1, self._num_cam+1):
                rgb = rgbs[cam_id]
                tracklet = labels[cam_id]
                for id, (x_min, y_min, x_max, y_max) in tracklet.items():
                    rgb = cv2.rectangle(rgb, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
                    rgb = cv2.putText(rgb, str(id), (int(x_min), int(y_min)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
                rgb = cv2.putText(rgb, str(frame_id), (30, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)
                if self._save_videos:
                    self._rgb_videos[cam_id].write(rgb)

if __name__ == '__main__':
    image_dir = 'path/to/image/folder'
    label_dir = 'path/to/label/folder'
    num_cam = 6
    start_frame, end_frame = 0, 3000
    data_reader = DataReader(image_dir, label_dir, num_cam, start_frame, end_frame)
    data_reader.plot()

