import os
import csv
import pandas as pd
import motmetrics as mm


class LabelReader:
    def __init__(self, label_dir) -> None:
        self._label_dir = label_dir
        self._num_frames = len([name for name in os.listdir(self._label_dir) if os.path.isfile(os.path.join(self._label_dir, name))])
    
    def read_single_frame(self, frame_id):
        tracklets = []
        with open(os.path.join(self._label_dir, 'topdown_'+str(frame_id).zfill(5)+'.csv'), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                tracklets.append({'FrameId':frame_id, 'Id':int(row[0]), 'X':int(float(row[2])), 'Y':int(float(row[1])), 'Confidence':1.0})
        return tracklets
    
    def read(self):
        rows_list = []
        for i in range(self._num_frames):
            rows_list.extend(self.read_single_frame(i))
        df = pd.DataFrame(rows_list)
        df = df.set_index(['FrameId', 'Id'])
        return df


if __name__ == '__main__':
    gt_df = LabelReader('/mnt/sdb/mmp_public/topdown_labels/63am/retail_0').read()
    pred_df = LabelReader('/mnt/sdb/mmp_public/topdown_labels/63am/retail_0').read()
    acc = mm.utils.compare_to_groundtruth(gt_df, pred_df, 'euc', distfields=['X', 'Y'], distth=525)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
