"""
Evaluation of -.tar.gz file.
Yu Fang - March 2019
"""

import os
import xml.dom.minidom

# from eval import eval
if os.path.exists("/mnt/localdata/Users/junlongli/projects/datasets/icdar2019"):
    PATH = "/mnt/localdata/Users/junlongli/projects/datasets/icdar2019/trackA_modern/test"
else:
    PATH = "/mnt/data/data/icdar2019/trackA_modern/test"
reg_gt_path = os.path.abspath(PATH)
reg_gt_path_archival = os.path.abspath(PATH)
reg_gt_path_modern = os.path.abspath(PATH)
str_gt_path_1 = os.path.abspath(PATH)
str_gt_path_2 = os.path.abspath(PATH)
str_gt_path_archival = os.path.abspath(PATH)
str_gt_path_modern = os.path.abspath(PATH)

import xml.dom.minidom
# from functools import cmp_to_key
from os.path import join as osj
from .data_structure import *


class eval:
    STR = "-str"
    REG = "-reg"
    DEFAULT_ENCODING = "UTF-8"
    # reg_gt_path = "./annotations/trackA/"
    # str_gt_path = "./annotations/trackB/"
    # reg_gt_path = os.path.abspath("data/test")
    # reg_gt_path_archival = os.path.abspath("data/test")
    # reg_gt_path_modern = os.path.abspath("data/test")
    # str_gt_path_1 = os.path.abspath("data/test")
    # str_gt_path_2 = os.path.abspath("data/test")
    # str_gt_path_archival = os.path.abspath("data/test")
    # str_gt_path_modern = os.path.abspath("data/test")

    # dummyDom = xml.dom.minidom.parse("./dummyXML.xml")

    def __init__(self, track, res_path):
        self.return_result = None
        self.reg = True
        self.str = False

        self.resultFile = res_path
        self.inPrefix = os.path.split(res_path)[-1].split(".")[0][:-7]

        if track == "-trackA":
            self.reg = True
            self.GTFile = osj(reg_gt_path, self.inPrefix + ".xml")
            # self.GTFile = osj(self.reg_gt_path, self.inPrefix)
        elif track == "-trackA1":  # archival documents
            self.reg = True
            self.GTFile = osj(reg_gt_path_archival, self.inPrefix + ".xml")
        elif track == "-trackA2":  # modern documents
            self.reg = True
            self.GTFile = osj(reg_gt_path_modern, self.inPrefix + ".xml")
        elif track == "-trackB1":
            self.str = True
            self.GTFile = osj(str_gt_path_1, self.inPrefix + ".xml")
            # self.GTFile = osj(self.str_gt_path_1, self.inPrefix)
        elif track == "-trackB2":
            self.str = True
            self.GTFile = osj(str_gt_path_2, self.inPrefix + ".xml")
            # print(self.GTFile)
            # self.GTFile = osj(self.str_gt_path_2, self.inPrefix)
        elif track == "-trackB2_a":
            self.str = True
            self.GTFile = osj(str_gt_path_archival, self.inPrefix + ".xml")
        elif track == "-trackB2_m":
            self.str = True
            self.GTFile = osj(str_gt_path_modern, self.inPrefix + ".xml")
        else:
            print(track)
            print("Not a valid track, please check your spelling.")

        # self.resultFile = res_path
        # self.inPrefix = os.path.split(res_path)[-1].split("-")[0]

        # if self.str:
        #     # self.GTFile = osj(self.str_gt_path, self.inPrefix + "-str.xml")
        #     self.GTFile = osj(self.str_gt_path, self.inPrefix + ".xml")
        # elif self.reg:
        #     # self.GTFile = osj(self.reg_gt_path, self.inPrefix + "-reg.xml")
        #     self.GTFile = osj(self.reg_gt_path, self.inPrefix + ".xml")
        # else:
        #     print("Not a valid track, please check your spelling.")

        self.gene_ret_lst()

    @property
    def result(self):
        return self.return_result

    def gene_ret_lst(self):
        ret_lst = []
        for iou in [0.6, 0.7, 0.8, 0.9]:
            temp = self.compute_retVal(iou)
            ret_lst.append(temp)
            # ret_lst.append(self.compute_retVal(iou))

        ret_lst.append(self.inPrefix + ".xml")
        # ret_lst.append(self.inPrefix)
        # print("Done processing {}\n".format(self.resultFile))
        self.return_result = ret_lst

    def compute_retVal(self, iou):
        gt_dom = xml.dom.minidom.parse(self.GTFile)
        # incorrect submission format handling
        try:
            result_dom = xml.dom.minidom.parse(self.resultFile)
        except Exception as e:
            # result_dom = xml.dom.minidom.parse(dummyDom)
            gt_tables = eval.get_table_list(gt_dom)
            retVal = ResultStructure(truePos=0, gtTotal=len(gt_tables), resTotal=0)
            return retVal

        # result_dom = xml.dom.minidom.parse(self.resultFile)
        if self.reg:
            ret = self.evaluate_result_reg(gt_dom, result_dom, iou)
            return ret
        if self.str:
            ret = self.evaluate_result_str(gt_dom, result_dom, iou)
            return ret

    @staticmethod
    def get_table_list(dom):
        """
        return a list of Table objects corresponding to the table element of the DOM.
        """
        return [Table(_nd) for _nd in dom.documentElement.getElementsByTagName("table")]

    @staticmethod
    def evaluate_result_reg(gt_dom, result_dom, iou_value):
        # parse the tables in input elements
        gt_tables = eval.get_table_list(gt_dom)
        result_tables = eval.get_table_list(result_dom)
        # duplicate result table list
        remaining_tables = result_tables.copy()

        # map the tables in gt and result file
        table_matches = []  # @param: table_matches - list of mapping of tables in gt and res file, in order (gt, res)
        for gtt in gt_tables:
            for rest in remaining_tables:
                if gtt.compute_table_iou(rest) >= iou_value:
                    remaining_tables.remove(rest)
                    table_matches.append((gtt, rest))
                    break

        assert len(table_matches) <= len(gt_tables)
        assert len(table_matches) <= len(result_tables)

        retVal = ResultStructure(truePos=len(table_matches), gtTotal=len(gt_tables), resTotal=len(result_tables))
        return retVal

    @staticmethod
    def evaluate_result_str(gt_dom, result_dom, iou_value, table_iou_value=0.8):
        # parse the tables in input elements
        gt_tables = eval.get_table_list(gt_dom)
        result_tables = eval.get_table_list(result_dom)

        # duplicate result table list
        remaining_tables = result_tables.copy()
        gt_remaining = gt_tables.copy()

        # map the tables in gt and result file
        table_matches = []  # @param: table_matches - list of mapping of tables in gt and res file, in order (gt, res)
        for gtt in gt_remaining:
            for rest in remaining_tables:
                # note: for structural analysis, use 0.8 for table mapping
                if gtt.compute_table_iou(rest) >= table_iou_value:
                    table_matches.append((gtt, rest))
                    remaining_tables.remove(rest)  # unsafe... should be ok with the break below
                    gt_remaining.remove(gtt)
                    break

        total_gt_relation, total_res_relation, total_correct_relation = 0, 0, 0
        for gt_table, ress_table in table_matches:

            # set up the cell mapping for matching tables
            cell_mapping = gt_table.find_cell_mapping(ress_table, iou_value)
            # set up the adj relations, convert the one for result table to a dictionary for faster searching
            gt_AR = gt_table.find_adj_relations()
            total_gt_relation += len(gt_AR)

            res_AR = ress_table.find_adj_relations()
            total_res_relation += len(res_AR)

            # Now map GT adjacency relations to result
            lMappedAR = []
            for ar in gt_AR:
                try:
                    resFromCell = cell_mapping[ar.fromText]
                    resToCell = cell_mapping[ar.toText]
                    # make a mapped adjacency relation
                    lMappedAR.append(AdjRelation(resFromCell, resToCell, ar.direction))
                except:
                    # no mapping is possible
                    pass

            # compare two list of adjacency relation
            correct_dect = 0
            for ar1 in res_AR:
                for ar2 in lMappedAR:
                    if ar1.isEqual(ar2):
                        correct_dect += 1
                        break

            total_correct_relation += correct_dect

        # handle gt_relations in unmatched gt table
        for gtt_remain in gt_remaining:
            total_gt_relation += len(gtt_remain.find_adj_relations())

        # handle gt_relation in unmatched res table
        for res_remain in remaining_tables:
            total_res_relation += len(res_remain.find_adj_relations())

        retVal = ResultStructure(truePos=total_correct_relation, gtTotal=total_gt_relation, resTotal=total_res_relation)
        return retVal

# calculate the gt adj_relations of the missing file
# @param: file_lst - list of missing ground truth file
# @param: cur_gt_num - current total of ground truth objects (tables / cells)
def process_missing_files(track, gt_file_lst, cur_gt_num):
    if track in ["-trackA", "-trackA1", "-trackA2"]:
        gt_file_lst_full = [osj(reg_gt_path, filename) for filename in gt_file_lst]
        for file in gt_file_lst_full:
            if os.path.split(file)[-1].split(".")[-1] == "xml":
                gt_dom = xml.dom.minidom.parse(file)
                gt_root = gt_dom.documentElement
                # tables = []
                table_elements = gt_root.getElementsByTagName("table")
                for res_table in table_elements:
                    # t = Table(res_table)
                    # tables.append(t)
                    cur_gt_num += 1
        return cur_gt_num
    elif track == "-trackB1":
        gt_file_lst_full = [osj(str_gt_path_1, filename) for filename in gt_file_lst]
        for file in gt_file_lst_full:
            if os.path.split(file)[-1].split(".")[-1] == "xml":
                gt_dom = xml.dom.minidom.parse(file)
                gt_root = gt_dom.documentElement
                tables = []
                table_elements = gt_root.getElementsByTagName("table")
                for res_table in table_elements:
                    t = Table(res_table)
                    tables.append(t)
                for table in tables:
                    cur_gt_num += len(table.find_adj_relations())
        return cur_gt_num
    elif track == "-trackB2":
        gt_file_lst_full = [osj(str_gt_path_2, filename) for filename in gt_file_lst]
        for file in gt_file_lst_full:
            if os.path.split(file)[-1].split(".")[-1] == "xml":
                gt_dom = xml.dom.minidom.parse(file)
                gt_root = gt_dom.documentElement
                tables = []
                table_elements = gt_root.getElementsByTagName("table")
                for res_table in table_elements:
                    t = Table(res_table)
                    tables.append(t)
                for table in tables:
                    cur_gt_num += len(table.find_adj_relations())
        return cur_gt_num

def calc(F1):
    sum_a = 0.6 * F1[0] + 0.7 * F1[1] + 0.8 * F1[2] + 0.9 * F1[3]
    sum_b = 0.6 + 0.7 + 0.8 + 0.9

    return sum_a / sum_b

def calc_table_score(result_path):
    # measure = eval(*sys.argv[1:])

    gt_file_lst = os.listdir(reg_gt_path_archival)
    track = "-trackA1"
    untar_path = result_path

    res_lst = []
    for root, files, dirs in os.walk(untar_path):
        for name in dirs:
            if name.split(".")[-1] == "xml":
                cur_filepath = osj(os.path.abspath(root), name)
                res_lst.append(eval(track, cur_filepath))
                # printing for debug
                # print("Processing... {}".format(name))
    # print("DONE WITH FILE PROCESSING\n")
    # note: results are stored as list of each when iou at [0.6, 0.7, 0.8, 0.9, gt_filename]
    # gt number should be the same for all files
    gt_num = 0
    correct_six, res_six = 0, 0
    correct_seven, res_seven = 0, 0
    correct_eight, res_eight = 0, 0
    correct_nine, res_nine = 0, 0


    for each_file in res_lst:
        # print(each_file)
        try:
            gt_file_lst.remove(each_file.result[-1])
            if each_file.result[-1].replace('.xml', '.jpg') in gt_file_lst:
                gt_file_lst.remove(each_file.result[-1].replace('.xml', '.jpg'))
            correct_six += each_file.result[0].truePos
            gt_num += each_file.result[0].gtTotal
            res_six += each_file.result[0].resTotal
            # print("{} {} {}".format(each_file.result[0].truePos, each_file.result[0].gtTotal, each_file.result[0].resTotal))

            correct_seven += each_file.result[1].truePos
            res_seven += each_file.result[1].resTotal

            correct_eight += each_file.result[2].truePos
            res_eight += each_file.result[2].resTotal

            correct_nine += each_file.result[3].truePos
            res_nine += each_file.result[3].resTotal
        except:
            print("Error occur in processing result list.")
            print(each_file.result[-1])
            break
            # print(each_file.result[-1])
            # print(each_file)

    # for file in gt_file_lst:
    #     if file.split(".") != "xml":
    #         gt_file_lst.remove(file)
    #     # print(gt_file_lst)

    for i in range(len(gt_file_lst) - 1, -1, -1):
        if gt_file_lst[i].split(".")[-1] != "xml":
            del gt_file_lst[i]

    if len(gt_file_lst) > 0:
        print("\nWarning: missing result annotations for file: {}\n".format(gt_file_lst))
        gt_total = process_missing_files(track, gt_file_lst, gt_num)
    else:
        gt_total = gt_num


    try:
        # print("Evaluation of {}".format(track.replace("-", "")))
        # iou @ 0.6
        p_six = correct_six / res_six
        r_six = correct_six / gt_total
        f1_six = 2 * p_six * r_six / (p_six + r_six)
        print("IOU @ 0.6 -\nprecision: {}\nrecall: {}\nf1: {}".format(p_six, r_six, f1_six))
        print("correct: {}, gt: {}, res: {}\n".format(correct_six, gt_total, res_six))

        # iou @ 0.7
        p_seven = correct_seven / res_seven
        r_seven = correct_seven / gt_total
        f1_seven = 2 * p_seven * r_seven / (p_seven + r_seven)
        print("IOU @ 0.7 -\nprecision: {}\nrecall: {}\nf1: {}".format(p_seven, r_seven, f1_seven))
        print("correct: {}, gt: {}, res: {}\n".format(correct_seven, gt_total, res_seven))

        # iou @ 0.8
        p_eight = correct_eight / res_eight
        r_eight = correct_eight / gt_total
        f1_eight = 2 * p_eight * r_eight / (p_eight + r_eight)
        print("IOU @ 0.8 -\nprecision: {}\nrecall: {}\nf1: {}".format(p_eight, r_eight, f1_eight))
        print("correct: {}, gt: {}, res: {}\n".format(correct_eight, gt_total, res_eight))

        # iou @ 0.9
        p_nine = correct_nine / res_nine
        r_nine = correct_nine / gt_total
        f1_nine = 2 * p_nine * r_nine / (p_nine + r_nine)
        print("IOU @ 0.9 -\nprecision: {}\nrecall: {}\nf1: {}".format(p_nine, r_nine, f1_nine))
        print("correct: {}, gt: {}, res: {}".format(correct_nine, gt_total, res_nine))

        F1 = [f1_six, f1_seven, f1_eight, f1_nine]
        wF1 = calc(F1)

        print("Average weight F1: {}".format(wF1))

        return {
            'p_six':p_six * 100,
            "r_six":r_six * 100,
            "f1_six":f1_six * 100,
            "p_seven":p_seven * 100,
            "r_seven":r_seven * 100,
            "f1_seven":f1_seven * 100,
            "p_eight":p_eight * 100,
            "r_eight":r_eight * 100,
            "f1_eight":f1_eight * 100,
            "p_nine":p_nine * 100,
            "r_nine":r_nine * 100,
            "f1_nine":f1_nine * 100,
            "wF1":wF1 * 100
        }
    except ZeroDivisionError:
        print(
            "Error: zero devision error found, (possible that no adjacency relations are found), please check the file input.")
        return {"wF1": 0}


if __name__=="__main__":
    pass
