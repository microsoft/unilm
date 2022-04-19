"""
Data structures used by the evaluation process.
Yu Fang - March 2019
"""

from collections import Iterable

import numpy as np
from shapely.geometry import Polygon


# helper functions
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item

# derived from https://blog.csdn.net/u012433049/article/details/82909484
def compute_poly_iou(list1, list2):
    a1 = np.array(list1, dtype=int).reshape(-1, 2)
    poly1 = Polygon(a1)
    poly1_clean = poly1.buffer(0)

    a2 = np.array(list2, dtype=int).reshape(-1, 2)
    poly2 = Polygon(a2)
    poly2_clean = poly2.buffer(0)

    try:
        # iou = poly1.intersection(poly2).area / poly1.union(poly2).area
        iou = poly1_clean.intersection(poly2_clean).area / poly1_clean.union(poly2_clean).area
    except ZeroDivisionError:
        iou = 0
    return iou


class Cell(object):
    # @:param start_row : start row index of the Cell
    # @:param start_col : start column index of the Cell
    # @:param end-row : end row index of the Cell
    # @:param end-col : end column index of the Cell
    # @:param cell_box: bounding-box of the Cell (coordinates are saved as a string)
    # @:param content_box: bounding-box of the text content within Cell (unused variable)
    # @:param cell_id: unique id of the Cell

    def __init__(self, table_id, start_row, start_col, cell_box, end_row, end_col, content_box=""):
        self._start_row = int(start_row)
        self._start_col = int(start_col)
        self._cell_box = cell_box
        self._content_box = content_box
        self._table_id = table_id    # the table_id this cell belongs to
        # self._cell_name = cell_id    # specify the cell using passed-in cell_id
        self._cell_id = id(self)
        # self._region = region

        # check for end-row and end-col special case
        if end_row == -1:
            self._end_row = self.start_row
        else:
            self._end_row = int(end_row)
        if end_col == -1:
            self._end_col = self._start_col
        else:
            self._end_col = int(end_col)

    @property
    def start_row(self):
        return self._start_row

    @property
    def start_col(self):
        return self._start_col

    @property
    def end_row(self):
        return self._end_row

    @property
    def end_col(self):
        return self._end_col

    @property
    def cell_box(self):
        return self._cell_box

    @property
    def content_box(self):
        return self._content_box

    @property
    def cell_id(self):
        return self._cell_id

    @property
    def table_id(self):
        return self._table_id

    def __str__(self):
        return "CELL row=[%d, %d] col=[%d, %d] (coords=%s)" %(self.start_row, self.end_row
                                                              , self.start_col, self.end_col
                                                              , self.cell_box)

    # return the IoU value of two cell blocks
    def compute_cell_iou(self, another_cell):
        cell_box_1_temp = []
        for el in self.cell_box.split():
            cell_box_1_temp.append((el.split(",")))
        cell_box_1 = list(flatten(cell_box_1_temp))
        cell_box_1 = [int(x) for x in cell_box_1]

        cell_box_2_temp = []
        for el in another_cell.cell_box.split():
            cell_box_2_temp.append((el.split(",")))
        cell_box_2 = list(flatten(cell_box_2_temp))
        cell_box_2 = [int(x) for x in cell_box_2]

        return compute_poly_iou(cell_box_1, cell_box_2)

    # check if the two cell object denotes same cell area in table
    def check_same(self, another_cell):
        return self._start_row == another_cell.start_row and self._end_row == another_cell.end_row and \
               self._start_col == another_cell.start_col and self._end_col == another_cell.end_col


# Note: currently save the relation with two cell object involved,
# can be replaced by cell_id in follow-up memory clean up
class AdjRelation:

    DIR_HORIZ = 1
    DIR_VERT = 2

    def __init__(self, fromText, toText, direction):
        # @param: fromText, toText are Cell objects （may be changed to cell-ID for further development）
        self._fromText = fromText
        self._toText = toText
        self._direction = direction

    @property
    def fromText(self):
        return self._fromText

    @property
    def toText(self):
        return self._toText

    @property
    def direction(self):
        return self._direction

    def __str__(self):
        if self.direction == self.DIR_VERT:
            dir = "vertical"
        else:
            dir = "horizontal"
        return 'ADJ_RELATION: ' + str(self._fromText) + '  ' + str(self._toText) + '    ' + dir

    def isEqual(self, otherRelation):
        return self.fromText.cell_id == otherRelation.fromText.cell_id and \
               self.toText.cell_id == otherRelation.toText.cell_id and self.direction == otherRelation.direction


class Table:

    def __init__(self, tableNode):
        self._root = tableNode
        self._id = id(self)
        self._table_coords = ""
        self._maxRow = 0    # PS: indexing from 0
        self._maxCol = 0
        self._cells = []    # save a table as list of <Cell>s
        self.adj_relations = []    # save the adj_relations for the table
        self.parsed = False
        self.found = False    # check if the find_adj_relations() has been called once

        self.parse_table()

    def __str__(self):
        return "TABLE object - {} row x {} col".format(self._maxRow+1, self._maxCol+1)

    @property
    def id(self):
        return self._id

    @property
    def table_coords(self):
        return self._table_coords

    @property
    def table_cells(self):
        return self._cells

    # parse input xml to cell lists
    def parse_table(self):
        # get the table bbox
        self._table_coords = str(self._root.getElementsByTagName("Coords")[0].getAttribute("points"))

        # get info for each cell
        cells = self._root.getElementsByTagName("cell")
        max_row = max_col = 0
        for cell in cells:
            sr = cell.getAttribute("start-row")
            sc = cell.getAttribute("start-col")
            cell_id = cell.getAttribute("id")
            b_points = str(cell.getElementsByTagName("Coords")[0].getAttribute("points"))
            # try:
            #     try:
            #         text = cell.getElementsByTagName("content")[0].firstChild.nodeValue
            #     except AttributeError:
            #         text = ""
            # except IndexError:
            #     text = "initialized cell as no content"
            er = cell.getAttribute("end-row") if cell.hasAttribute("end-row") else -1
            ec = cell.getAttribute("end-col") if cell.hasAttribute("end-col") else -1
            new_cell = Cell(table_id=str(self.id), start_row=sr, start_col=sc, cell_box=b_points,
                            end_row=er, end_col=ec)
            max_row = max(max_row, int(sr), int(er))
            max_col = max(max_col, int(sc), int(ec))
            self._cells.append(new_cell)
        self._maxCol = max_col
        self._maxRow = max_row
        self.parsed = True

    # generate a table-like structure for finding adj_relations
    def convert_2d(self):
        table = [[0 for x in range(self._maxCol+1)] for y in range(self._maxRow+1)]    # init blank cell with int 0
        for cell in self._cells:
            cur_row = cell.start_row
            while cur_row <= cell.end_row:
                cur_col = cell.start_col
                while cur_col <= cell.end_col:
                    temp = table[cur_row][cur_col]
                    if temp == 0:
                        table[cur_row][cur_col] = cell
                    elif type(temp) == list:
                        temp.append(cell)
                        table[cur_row][cur_col] = temp
                    else:
                        table[cur_row][cur_col] = [temp, cell]
                    cur_col += 1
                cur_row += 1

        return table

    def find_adj_relations(self):
        if self.found:
            return self.adj_relations
        else:
            # if len(self._cells) == 0:
            if self.parsed == False:
                # fix: cases where there's no cell in table?
                print("table is not parsed for further steps.")
                self.parse_table()
                self.find_adj_relations()
            else:
                retVal = []
                tab = self.convert_2d()

                # find horizontal relations
                for r in range(self._maxRow+1):
                    for c_from in range(self._maxCol):
                        temp_pos = tab[r][c_from]
                        if temp_pos == 0:
                            continue
                        elif type(temp_pos) == list:
                            for cell in temp_pos:
                                c_to = c_from + 1
                                if tab[r][c_to] != 0:
                                    # find relation between two adjacent cells
                                    if type(tab[r][c_to]) == list:
                                        for cell_to in tab[r][c_to]:
                                            if cell != cell_to and (not cell.check_same(cell_to)):
                                                adj_relation = AdjRelation(cell, cell_to, AdjRelation.DIR_HORIZ)
                                                retVal.append(adj_relation)
                                    else:
                                        if cell != tab[r][c_to]:
                                            adj_relation = AdjRelation(cell, tab[r][c_to], AdjRelation.DIR_HORIZ)
                                            retVal.append(adj_relation)
                                else:
                                    # find the next non-blank cell, if exists
                                    for temp in range(c_from + 1, self._maxCol + 1):
                                        if tab[r][temp] != 0:
                                            if type(tab[r][temp]) == list:
                                                for cell_to in tab[r][temp]:
                                                    adj_relation = AdjRelation(cell, cell_to,
                                                                               AdjRelation.DIR_HORIZ)
                                                    retVal.append(adj_relation)
                                            else:
                                                adj_relation = AdjRelation(cell, tab[r][temp],
                                                                           AdjRelation.DIR_HORIZ)
                                                retVal.append(adj_relation)
                                            break
                        else:
                            c_to = c_from + 1
                            if tab[r][c_to] != 0:
                                # find relation between two adjacent cells
                                if type(tab[r][c_to]) == list:
                                    for cell_to in tab[r][c_to]:
                                        if temp_pos != cell_to:
                                            adj_relation = AdjRelation(temp_pos, cell_to, AdjRelation.DIR_HORIZ)
                                            retVal.append(adj_relation)
                                else:
                                    if temp_pos != tab[r][c_to]:
                                        adj_relation = AdjRelation(temp_pos, tab[r][c_to], AdjRelation.DIR_HORIZ)
                                        retVal.append(adj_relation)
                            else:
                                # find the next non-blank cell, if exists
                                for temp in range(c_from + 1, self._maxCol + 1):
                                    if tab[r][temp] != 0:
                                        if type(tab[r][temp]) == list:
                                            for cell_to in tab[r][temp]:
                                                adj_relation = AdjRelation(temp_pos, cell_to,
                                                                           AdjRelation.DIR_HORIZ)
                                                retVal.append(adj_relation)
                                        else:
                                            adj_relation = AdjRelation(temp_pos, tab[r][temp], AdjRelation.DIR_HORIZ)
                                            retVal.append(adj_relation)
                                        break

                # find vertical relations
                for c in range(self._maxCol+1):
                    for r_from in range(self._maxRow):
                        temp_pos = tab[r_from][c]
                        if temp_pos == 0:
                            continue
                        elif type(temp_pos) == list:
                            for cell in temp_pos:
                                r_to = r_from + 1
                                if tab[r_to][c] != 0:
                                    # find relation between two adjacent cells
                                    if type(tab[r_to][c]) == list:
                                        for cell_to in tab[r_to][c]:
                                            if cell != cell_to and (not cell.check_same(cell_to)):
                                                adj_relation = AdjRelation(cell, cell_to, AdjRelation.DIR_VERT)
                                                retVal.append(adj_relation)
                                    else:
                                        if cell != tab[r_to][c]:
                                            adj_relation = AdjRelation(cell, tab[r_to][c], AdjRelation.DIR_VERT)
                                            retVal.append(adj_relation)
                                else:
                                    # find the next non-blank cell, if exists
                                    for temp in range(r_from + 1, self._maxRow + 1):
                                        if tab[temp][c] != 0:
                                            if type(tab[temp][c]) == list:
                                                for cell_to in tab[temp][c]:
                                                    adj_relation = AdjRelation(cell, cell_to,
                                                                               AdjRelation.DIR_VERT)
                                                    retVal.append(adj_relation)
                                            else:
                                                adj_relation = AdjRelation(cell, tab[temp][c],
                                                                           AdjRelation.DIR_VERT)
                                                retVal.append(adj_relation)
                                            break
                        else:
                            r_to = r_from + 1
                            if tab[r_to][c] != 0:
                                # find relation between two adjacent cells
                                if type(tab[r_to][c]) == list:
                                    for cell_to in tab[r_to][c]:
                                        if temp_pos != cell_to:
                                            adj_relation = AdjRelation(temp_pos, cell_to, AdjRelation.DIR_VERT)
                                            retVal.append(adj_relation)
                                else:
                                    if temp_pos != tab[r_to][c]:
                                        adj_relation = AdjRelation(temp_pos, tab[r_to][c], AdjRelation.DIR_VERT)
                                        retVal.append(adj_relation)
                            else:
                                # find the next non-blank cell, if exists
                                for temp in range(r_from + 1, self._maxRow + 1):
                                    if tab[temp][c] != 0:
                                        if type(tab[temp][c]) == list:
                                            for cell_to in tab[temp][c]:
                                                adj_relation = AdjRelation(temp_pos, cell_to, AdjRelation.DIR_VERT)
                                                retVal.append(adj_relation)
                                        else:
                                            adj_relation = AdjRelation(temp_pos, tab[temp][c], AdjRelation.DIR_VERT)
                                            retVal.append(adj_relation)
                                        break

                # eliminate duplicates
                repeat = True
                while repeat:
                    repeat = False
                    duplicates = []

                    for ar1 in retVal:
                        for ar2 in retVal:
                            if ar1 != ar2:
                                if ar1.direction == ar2.direction and ar1.fromText == ar2.fromText and\
                                        ar1.toText == ar2.toText:
                                    duplicates.append(ar2)
                                    break
                        else:
                            continue
                        break

                    if len(duplicates) > 0:
                        repeat = True
                        retVal.remove(duplicates[0])

                self.found = True
                self.adj_relations = retVal
            return self.adj_relations

    # compute the IOU of table, pass-in var is another Table object
    def compute_table_iou(self, another_table):
        table_box_1_temp = []
        for el in self.table_coords.split():
            table_box_1_temp.append((el.split(",")))
        table_box_1 = list(flatten(table_box_1_temp))
        table_box_1 = [int(x) for x in table_box_1]

        table_box_2_temp = []
        for el in another_table.table_coords.split():
            table_box_2_temp.append((el.split(",")))
        table_box_2 = list(flatten(table_box_2_temp))
        table_box_2 = [int(x) for x in table_box_2]

        return compute_poly_iou(table_box_1, table_box_2)

    # find the cell mapping of tables as dictionary, pass-in var is another table and the desired IOU value
    def find_cell_mapping(self, target_table, iou_value):
        mapped_cell = []    # store the matches as tuples - (gt, result) mind the order of table when passing in
        for cell_1 in self.table_cells:
            for cell_2 in target_table.table_cells:
                if cell_1.compute_cell_iou(cell_2) >= iou_value:
                    mapped_cell.append((cell_1, cell_2))
                    break
        ret = dict(mapped_cell)
        # print(ret)
        return ret

    # to print a table cell mapping
    @classmethod
    def printCellMapping(cls, dMappedCell):
        print("-"*25)
        for cell1, cell2 in dMappedCell.items():
            print("  ", cell1, " --> ", cell2)

    # to print a table set of adjacency relations
    @classmethod
    def printAdjacencyRelationList(cls, lAdjRel, title=""):
        print("--- %s "%title + "-"*25)
        for adj in lAdjRel:
            print(adj)


class ResultStructure:

    def __init__(self, truePos, gtTotal, resTotal):
        self._truePos = truePos
        self._gtTotal = gtTotal
        self._resTotal = resTotal

    @property
    def truePos(self):
        return self._truePos

    @property
    def gtTotal(self):
        return self._gtTotal

    @property
    def resTotal(self):
        return self._resTotal

    def __str__(self):
        return "true: {}, gt: {}, res: {}".format(self._truePos, self._gtTotal, self._resTotal)