import os
import json
import copy
import itertools
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.evaluation import COCOEvaluator

from .concern.icdar2015_eval.detection.iou import DetectionIoUEvaluator

class FUNSDEvaluator(COCOEvaluator):
    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._logger.warning("[evaluating...]The evaluator may take long time")

        id2img = {}
        gt = {}
        with open('data/instances_test.json', 'r',
                  encoding='utf-8') as fr:
            data = json.load(fr)
            for img in data['images']:
                id = img['id']
                name = os.path.basename(img['file_name'])[:-len('.jpg')]
                assert id not in id2img.keys()
                id2img[id] = name
            assert len(id2img) == len(data['images'])

            img2id, id2bbox = {}, {}
            for i in range(len(data['images'])):
                key = os.path.basename(data['images'][i]['file_name'][:-len('.png')])
                assert key not in img2id.keys()
                img2id[key] = data['images'][i]['id']
            for i in range(len(data['annotations'])):
                img_id = data['annotations'][i]['image_id']
                if img_id not in id2bbox.keys():
                    id2bbox[img_id] = []
                x0, y0, w, h = data['annotations'][i]['bbox']
                x1, y1 = x0 + w, y0 + h
                line = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                id2bbox[img_id].append(
                    {
                        'points': line,
                        'text': 1234,
                        'ignore': False,
                    }
                )
            for key, val in img2id.items():
                assert key not in gt.keys()
                gt[key] = id2bbox[val]

        self._results = OrderedDict()

        evaluator = DetectionIoUEvaluator()

        for iter in range(3, 10):
            thr = iter * 0.1
            self._results[thr] = {}

            total_prediction = {}
            for cur_pred in predictions:
                assert cur_pred['image_id'] in id2img.keys()
                id = id2img[cur_pred['image_id']]
                if id not in total_prediction.keys(): total_prediction[id] = []

                for cur_inst in cur_pred['instances']:
                    x0, y0, w, h = cur_inst['bbox']
                    cur_score = cur_inst['score']
                    if cur_score < thr:
                        continue

                    x1, y1 = x0 + w, y0 + h

                    x0, x1 = int(x0 + 0.5), int(x1 + 0.5)
                    y0, y1 = int(y0 + 0.5), int(y1 + 0.5)

                    min_x, max_x = min([x0, x1]), max([x0, x1])
                    min_y, max_y = min([y0, y1]), max([y0, y1])

                    pred_line = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
                    pred_line_str = ','.join(list(map(str, pred_line)))

                    total_prediction[id].append(pred_line_str)

            final_gt = []
            final_res = []
            for key, _ in gt.items():
                final_gt.append(copy.deepcopy(gt[key]))

                cur_res = []
                pred = total_prediction[key]
                for i in range(len(pred)):
                    line = list(map(int, pred[i].split(',')))
                    line = [(line[0], line[1]), (line[2], line[3]), (line[4], line[5]), (line[6], line[7])]
                    cur_res.append(
                        {
                            'points': line,
                            'text': 1234,
                            'ignore': False,
                        }
                    )
                final_res.append(cur_res)

            results = []
            for cur_gt, pred in zip(final_gt, final_res):
                results.append(evaluator.evaluate_image(cur_gt, pred))
            metrics = evaluator.combine_results(results)
            for key, val in metrics.items():
                self._results["{:.1f}_{}".format(thr, key)] = val

        return copy.deepcopy(self._results)
