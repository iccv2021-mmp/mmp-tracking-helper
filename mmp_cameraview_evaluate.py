import os
import json
from collections import defaultdict, OrderedDict
import pandas as pd
import motmetrics as mm


class LabelReader:
    def __init__(self, label_dir, num_cam) -> None:
        self._label_dir = label_dir
        self._num_frames = len([name for name in os.listdir(self._label_dir) if os.path.isfile(os.path.join(self._label_dir, name))]) // num_cam
        self._num_cam = num_cam
    
    def read_single_frame(self, frame_id):
        labels = {}
        for cam_id in range(1, self._num_cam+1):
            raw_labels = json.load(open(os.path.join(self._label_dir, 'rgb_'+str(frame_id).zfill(5)+'_'+str(cam_id)+'.json')))
            raw_list = []
            for id, (x_min, y_min, x_max, y_max) in raw_labels.items():
                raw_list.append({'FrameId':frame_id, 'Id':int(id), 'X':int(float(x_min)), 'Y':int(float(y_min)), 'Width':int(float(x_max-x_min)), 'Height':int(float(y_max-y_min)), 'Confidence':1.0})
            labels[cam_id] = raw_list
        return labels
    
    def read(self):
        rows_list = defaultdict(list)
        for i in range(self._num_frames):
            single_frame_labels = self.read_single_frame(i)
            for cam_id in range(1, self._num_cam+1):
                rows_list[cam_id].extend(single_frame_labels[cam_id])
        return OrderedDict([(cam_id, pd.DataFrame(rows).set_index(['FrameId', 'Id'])) for cam_id, rows in rows_list.items()])


class PredReader:
    def __init__(self, label_dir, num_cam) -> None:
        self._label_dir = label_dir
        self._num_cam = num_cam
    
    def read_txt(self, filename):
        results = []
        with open(filename, 'r') as infile:
            lines = infile.readlines()
            for line in lines:
                line = line.split(',')
                results.append({'FrameId':int(line[0]), 'Id':int(line[1]), 'X':int(line[2]), 'Y':int(line[3]), 'Width':int(line[4]), 'Height':int(line[5]), 'Confidence':float(line[6])})
        return results
    
    def read(self):
        rows_list = defaultdict(list)
        for cam_id in range(1, self._num_cam+1):
            rows_list[cam_id].extend(self.read_txt(os.path.join(self._label_dir, 'retail_0_'+str(cam_id)+'.txt')))
        return OrderedDict([(cam_id, pd.DataFrame(rows).set_index(['FrameId', 'Id'])) for cam_id, rows in rows_list.items()])


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
        names.append(k)
    return accs, names


if __name__ == '__main__':
    gt_dfs = LabelReader('/mnt/sdb/mmp_public/labels/63am/retail_0', 6).read()
    # pred_dfs = LabelReader('/mnt/sdb/mmp_public/labels/63am/retail_0', 6).read()
    pred_dfs = PredReader('/home/xthan/Downloads/bbox', 6).read()
    accs, names = compare_dataframes(gt_dfs, pred_dfs)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
