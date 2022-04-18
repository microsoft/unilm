import copy
import itertools
import os
import os.path as osp
import shutil
from collections import OrderedDict
from xml.dom.minidom import Document

import detectron2.utils.comm as comm
import torch
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.file_io import PathManager

from .table_evaluation.evaluate import calc_table_score


class ICDAREvaluator(COCOEvaluator):
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

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
            self.evaluate_table(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def evaluate_table(self, predictions):
        xml_dir = self.convert_to_xml(predictions)
        results = calc_table_score(xml_dir)
        self._results["wF1"] = results['wF1']

    def convert_to_xml(self, predictions):
        output_dir = osp.join(self._output_dir, "xml_results")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        results_dict = {}
        for result in coco_results:
            if result["score"] < 0.7:
                continue
            image_id = result["image_id"]
            if image_id not in results_dict:
                results_dict[image_id] = []

            results_dict[image_id].append(result)

        for image_id, tables in results_dict.items():
            file_name = f"cTDaR_t{image_id:05d}.jpg"
            doc = Document()
            root = doc.createElement('document')
            root.setAttribute('filename', file_name)
            doc.appendChild(root)
            for table_id, table in enumerate(tables, start=1):
                nodeManager = doc.createElement('table')
                nodeManager.setAttribute('id', str(table_id))
                bbox = list(map(int, table['bbox']))
                bbox_str = '{},{} {},{} {},{} {},{}'.format(bbox[0], bbox[1],
                                                            bbox[0], bbox[1] + bbox[3],
                                                            bbox[0] + bbox[2], bbox[1] + bbox[3],
                                                            bbox[0] + bbox[2], bbox[1])
                nodeCoords = doc.createElement('Coords')
                nodeCoords.setAttribute('points', bbox_str)
                nodeManager.appendChild(nodeCoords)
                root.appendChild(nodeManager)
            filename = '{}-result.xml'.format(file_name[:-4])
            fp = open(os.path.join(output_dir, filename), 'w')
            doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
            fp.flush()
            fp.close()
        return output_dir


if __name__ == '__main__':
    pass
