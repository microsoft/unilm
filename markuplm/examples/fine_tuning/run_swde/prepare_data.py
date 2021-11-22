# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extracting XPaths of the values of all fields for SWDE dataset."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import collections
import os
import pickle
import random
import re
import sys
import unicodedata

from absl import app
from absl import flags
import lxml
from lxml import etree
from lxml.html.clean import Cleaner
from tqdm import tqdm

import constants
import multiprocessing as mp

FLAGS = flags.FLAGS
random.seed(42)

flags.DEFINE_integer("n_pages", 2000, "The maximum number of pages to read.")
flags.DEFINE_string(
    "input_groundtruth_path", "",
    "The root path to parent folder of all ground truth files.")
flags.DEFINE_string("input_pickle_path", "",
                    "The root path to pickle file of swde html content.")
flags.DEFINE_string(
    "output_data_path", "",
    "The path of the output file containing both the input sequences and "
    "output sequences of the sequence tagging version of swde dataset.")


def clean_spaces(text):
    r"""Clean extra spaces in a string.

    Example:
      input: " asd  qwe   " --> output: "asd qwe"
      input: " asd\t qwe   " --> output: "asd qwe"
    Args:
      text: the input string with potentially extra spaces.

    Returns:
      a string containing only the necessary spaces.
    """
    return " ".join(re.split(r"\s+", text.strip()))


def clean_format_str(text):
    """Cleans unicode control symbols, non-ascii chars, and extra blanks."""
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = "".join([c if ord(c) < 128 else "" for c in text])
    text = clean_spaces(text)
    return text


def non_ascii_equal(website, field, value, node_text):
    """Compares value and node_text by their non-ascii texts.

    Website/field are used for handling special cases.

    Args:
      website: the website that the value belongs to, used for dealing with
        special cases.
      field: the field that the value belongs to, used for dealing with special
        cases.
      value: the value string that we want to compare.
      node_text: the clean text of the node that we want to compare.

    Returns:
      a boolean variable indicating if the value and node_text are equal.
    """
    value = clean_format_str(value)
    node_text = clean_format_str(node_text)

    # A special case in the ALLMOVIE website's MPAA_RATING,
    # the truth values are not complete but only the first character.
    # For example, truth value in the file:"P", which should be "PG13" in htmls.
    # Note that the length of the truth should be less than 5.
    if website == "allmovie" and field == "mpaa_rating" and len(node_text) <= 5:
        return node_text.strip().startswith(value.strip())

    # A special case in the AMCTV website, DIRECTOR field.
    # The name are not complete in the truth values.
    # E.g. truth value in files, "Roy Hill" and real value: "Geogre Roy Hill".
    if website == "amctv" and field == "director":
        return node_text.strip().endswith(value.strip())
    return value.strip() == node_text.strip()


def match_value_node(node, node_text, current_xpath_data, overall_xpath_dict,
                     text_part_flag, groundtruth_value, matched_xpaths, website,
                     field, dom_tree, current_page_nodes_in_order, is_truth_value_list):
    """Matches the ground truth value with a specific node in the domtree.

    In the function, the current_xpath_data, overall_xpath_dict, matched_xpaths
    will be updated.

    Args:
      is_truth_value_list: [], indicate which node is the truth-value
      current_page_nodes_in_order: [(text, xpath)] seq
      node: the node on the domtree that we are going to match.
      node_text: the text inside this node.
      current_xpath_data: the dictionary of the xpaths of the current domtree.
      overall_xpath_dict: the dictionary of the xpaths of the current website.
      text_part_flag: to match the "text" or the "tail" part of the node.
      groundtruth_value: the value of our interest to match.
      matched_xpaths: the existing matched xpaths list for this value on domtree.
      website: the website where the value is from.
      field: the field where the value is from.
      dom_tree: the current domtree object, used for getting paths.
    """
    assert text_part_flag in ["node_text", "node_tail_text"]
    # Dealing with the cases with multiple <br>s in the node text,
    # where we need to split and create new tags of matched_xpaths.
    # For example, "<div><span>asd<br/>qwe</span></div>"
    len_brs = len(node_text.split("--BRRB--"))  # The number of the <br>s.
    for index, etext in enumerate(node_text.split("--BRRB--")):
        if text_part_flag == "node_text":
            xpath = dom_tree.getpath(node)
        elif text_part_flag == "node_tail_text":
            xpath = dom_tree.getpath(node) + "/tail"
        if len_brs >= 2:
            xpath += "/br[%d]" % (index + 1)  # E.g. /div/span/br[1]
        clean_etext = clean_spaces(etext)

        # Update the dictionary.
        current_xpath_data[xpath] = clean_etext
        overall_xpath_dict[xpath].add(clean_etext)
        current_page_nodes_in_order.append((clean_etext, xpath))

        # Exactly match the text.
        if non_ascii_equal(website, field, groundtruth_value, clean_etext):
            matched_xpaths.append(xpath)
            is_truth_value_list.append(len(current_page_nodes_in_order) - 1)

        # 这里我们更新三样东西
        # 如果当前节点与truth_value一致，则将当前xpath加入matched_xpaths
        # 此外，还需要 current_xpath_data[xpath] = clean_etext,即记录当前页面 该xpath对应的文字
        # 以及 overall_xpath_dict[xpath].add(clean_etext)，即记录当前网址上该xpath对应的文字，以add加入集合


def get_value_xpaths(dom_tree,
                     truth_value,
                     overall_xpath_dict,
                     website="",
                     field=""):
    """Gets a list of xpaths that contain a text truth_value in DOMTree objects.

    Args:
      dom_tree: the DOMTree object of a specific HTML page.
      truth_value: a certain groundtruth value.
      overall_xpath_dict: a dict maintaining all xpaths data of a website.
      website: the website name.
      field: the field name.

    Returns:
      xpaths: a list of xpaths containing the truth_value exactly as inner texts.
      current_xpath_data: the xpaths and corresponding values in this DOMTree.
    """
    if not truth_value:
        #  Some values are empty strings, that are not in the DOMTree.
        return []

    xpaths = []  # The resulting list of xpaths to be returned.
    current_xpath_data = dict()  # The resulting dictionary to save all page data.

    current_page_nodes_in_order = []
    is_truth_value_list = []

    # Some values contains HTML tags and special strings like "&nbsp;"
    # So we need to escape the HTML by parsing and then extract the inner text.
    value_dom = lxml.html.fromstring(truth_value)
    value = " ".join(etree.XPath("//text()")(value_dom))
    value = clean_spaces(value)

    # Iterate all the nodes in the given DOMTree.
    for e in dom_tree.iter():
        # The value can only be matched in the text of the node or the tail.
        if e.text:
            match_value_node(
                e,
                e.text,
                current_xpath_data,
                overall_xpath_dict,
                text_part_flag="node_text",
                groundtruth_value=value,
                matched_xpaths=xpaths,
                website=website,
                field=field,
                dom_tree=dom_tree,
                current_page_nodes_in_order=current_page_nodes_in_order,
                is_truth_value_list=is_truth_value_list
            )
        if e.tail:
            match_value_node(
                e,
                e.tail,
                current_xpath_data,
                overall_xpath_dict,
                text_part_flag="node_tail_text",
                groundtruth_value=value,
                matched_xpaths=xpaths,
                website=website,
                field=field,
                dom_tree=dom_tree,
                current_page_nodes_in_order=current_page_nodes_in_order,
                is_truth_value_list=is_truth_value_list
            )

    return xpaths, current_xpath_data, current_page_nodes_in_order, is_truth_value_list


def get_dom_tree(html, website):
    """Parses a HTML string to a DOMTree.

    We preprocess the html string and use lxml lib to get a tree structure object.

    Args:
      html: the string of the HTML document.
      website: the website name for dealing with special cases.

    Returns:
      A parsed DOMTree object using lxml library.
    """
    cleaner = Cleaner()
    cleaner.javascript = True
    cleaner.style = True
    cleaner.page_structure = False
    html = html.replace("\0", "")  # Delete NULL bytes.
    # Replace the <br> tags with a special token for post-processing the xpaths.
    html = html.replace("<br>", "--BRRB--")
    html = html.replace("<br/>", "--BRRB--")
    html = html.replace("<br />", "--BRRB--")
    html = html.replace("<BR>", "--BRRB--")
    html = html.replace("<BR/>", "--BRRB--")
    html = html.replace("<BR />", "--BRRB--")

    # A special case in this website, where the values are inside the comments.
    if website == "careerbuilder":
        html = html.replace("<!--<tr>", "<tr>")
        html = html.replace("<!-- <tr>", "<tr>")
        html = html.replace("<!--  <tr>", "<tr>")
        html = html.replace("<!--   <tr>", "<tr>")
        html = html.replace("</tr>-->", "</tr>")

    html = clean_format_str(html)
    x = lxml.html.fromstring(html)
    etree_root = cleaner.clean_html(x)
    dom_tree = etree.ElementTree(etree_root)
    return dom_tree


def load_html_and_groundtruth(vertical_to_load, website_to_load):
    """
    DONE READ!
    """
    # example is `book` and `abebooks`
    """Loads and returns the html sting and ground turth data as a dictionary."""
    all_data_dict = collections.defaultdict(dict)
    vertical_to_websites_map = constants.VERTICAL_WEBSITES
    gt_path = FLAGS.input_groundtruth_path

    """
    First build groudtruth dict
    """
    for v in vertical_to_websites_map:
        if v != vertical_to_load: continue
        for truthfile in os.listdir(os.path.join(gt_path, v)):
            # For example, a groundtruth file name can be "auto-yahoo-price.txt".
            vertical, website, field = truthfile.replace(".txt", "").split("-")
            # like book , amazon , isbn_13

            if website != website_to_load:
                continue

            with open(os.path.join(gt_path, v, truthfile), "r") as gfo:
                lines = gfo.readlines()
                for line in lines[2:]:
                    # Each line should contains more than 3 elements splitted by \t
                    # which are: index, number of values, value1, value2, etc.
                    item = line.strip().split("\t")
                    index = item[0]  # like 0123
                    num_values = int(item[1])  # Can be 0 (when item[2] is "<NULL>").
                    all_data_dict[index]["field-" + field] = dict(values=item[2:2 + num_values])
            # {"0123":
            #   {"field-engine":
            #       {"values":["engine A","engine B"]},
            #    "field-price":
            #   }
            # }
    """

    this is an example for book-abebooks-0000.htm
    <-- all_data_dict["0000"] -->
    {
        'field-publication_date': {'values': ['2008']}, 
        'field-author': {'values': ['Howard Zinn', 'Paul Buhle', 'Mike Konopacki']}, 
        'field-title': {'values': ["A People's History of American Empire"]}, 
        'field-publisher': {'values': ['Metropolitan Books']}, 
        'field-isbn_13': {'values': ['9780805087444']}
    }

    """

    print("Reading the pickle of SWDE original dataset.....", file=sys.stderr)
    with open(FLAGS.input_pickle_path, "rb") as gfo:
        swde_html_data = pickle.load(gfo)
    # {"vertical":'book',"website":'book-amazon(2000)',"path:'book/book-amazon(2000)/0000.htm',"html_str":xx} here

    for page in tqdm(swde_html_data, desc="Loading HTML data"):
        vertical = page["vertical"]
        website = page["website"]
        website = website[website.find("-") + 1:website.find("(")]

        if vertical != vertical_to_load or website != website_to_load:
            continue

        path = page["path"]  # For example, auto/auto-aol(2000)/0000.htm
        html_str = page["html_str"]
        _, _, index = path.split("/")  # website be like auto-aol(2000)
        index = index.replace(".htm", "")

        all_data_dict[index]["html_str"] = html_str
        all_data_dict[index]["path"] = path

    """
        this is an example for book-abebooks-0000.htm
        <-- all_data_dict["0000"] -->
        {
            'field-publication_date': {'values': ['2008']}, 
            'field-author': {'values': ['Howard Zinn', 'Paul Buhle', 'Mike Konopacki']}, 
            'field-title': {'values': ["A People's History of American Empire"]}, 
            'field-publisher': {'values': ['Metropolitan Books']}, 
            'field-isbn_13': {'values': ['9780805087444']},
            'path': 'book/book-abebooks(2000)/0000.htm',
            'html_str': omitted,
        }
    """

    # all_data_dict here has all the pages
    # however, only those in swde.pickle has the newly-appended 'path' and 'html_str'

    return all_data_dict


def get_field_xpaths(all_data_dict,
                     vertical_to_process,
                     website_to_process,
                     n_pages,
                     max_variable_nodes_per_website):
    """Gets xpaths data for each page in the data dictionary.

    Args:
      all_data_dict: the dictionary saving both the html content and the truth.
      vertical_to_process: the vertical that we are working on;
      website_to_process: the website that we are working on.
      n_pages: we will work on the first n_pages number of the all pages.
      max_variable_nodes_per_website: top N frequent variable nodes as the final set.
    """
    # Saving the xpath info of the whole website,
    #  - Key is a xpath.
    #  - Value is a set of text appeared before inside the node.
    overall_xpath_dict = collections.defaultdict(set)
    #  Update page data with groundtruth xpaths and the overall xpath-value dict.
    for index in tqdm(all_data_dict, desc="Processing %s" % website_to_process, total=n_pages):
        if int(index) >= n_pages:
            continue
        # We add dom-tree attributes for the first n_pages
        page_data = all_data_dict[index]
        html = page_data["html_str"]
        dom_tree = get_dom_tree(html, website=website_to_process)
        page_data["dom_tree"] = dom_tree

        # Match values of each field for the current page.
        for field in page_data:
            if not field.startswith("field-"):
                continue

            # Saving the xpaths of the values for each field.
            page_data[field]["groundtruth_xpaths"] = set()
            page_data[field]["is_truth_value_list"] = set()

            for value in page_data[field]["values"]:
                xpaths, current_xpath_data, current_page_nodes_in_order, is_truth_value_list = \
                    get_value_xpaths(dom_tree,
                                     value,
                                     overall_xpath_dict,
                                     website_to_process,
                                     field[6:])

                # Assert each truth value can be founded in >=1 nodes.
                assert len(xpaths) >= 1, \
                    "%s;\t%s;\t%s;\t%s; is not found" % (website_to_process, field, index, value)

                # Update the page-level xpath information.
                page_data[field]["groundtruth_xpaths"].update(xpaths)
                page_data[field]["is_truth_value_list"].update(is_truth_value_list)

            # now for each page_data
            # an example
            # page_data["field-author"] =
            # {
            #   'values': ['Dave Kemper', 'Patrick Sebranek', 'Verne Meyer'],
            #   'groundtruth_xpaths':
            #       {'/html/body/div[2]/div[2]/div[2]/div[1]/h3/a[3]',
            #        '/html/body/div[2]/div[2]/div[2]/div[1]/h3/a[2]',
            #        '/html/body/div[2]/div[2]/div[2]/div[1]/h3/a[1]',
            #        '/html/body/div[2]/div[2]/div[3]/div[3]/p/a'}
            # }

        page_data["xpath_data"] = current_xpath_data  #
        page_data["doc_strings"] = current_page_nodes_in_order  # [(text, xpath)*N]
        # page_data["reversed_doc_strings_ids"] = {v[0]: i for i, v in enumerate(current_page_nodes_in_order)}

    # page_data["doc_strings"] is the basis of our transformers-based method!!!

    # Define the fixed-text nodes and variable nodes.
    fixed_nodes = set()
    variable_nodes = set()
    # 这里对这个网址上的所有xpath进行排序
    # 以对应的不同文本数目倒序排列
    node_variability = sorted(
        [(xpath, len(text_set)) for xpath, text_set in overall_xpath_dict.items()],
        key=lambda x: x[1],
        reverse=True
    )

    for xpath, variability in node_variability:
        # variability 为xpath的可变性
        if variability > 5 and len(variable_nodes) < max_variable_nodes_per_website:
            variable_nodes.add(xpath)
        else:
            fixed_nodes.add(xpath)

    print("Vertical: %s; Website: %s; fixed_nodes: %d; variable_nodes: %d" %
          (
              vertical_to_process, website_to_process, len(fixed_nodes), len(variable_nodes)
          )
          )

    assure_value_variable(all_data_dict, variable_nodes, fixed_nodes, n_pages)
    all_data_dict["fixed_nodes"] = list(fixed_nodes)
    all_data_dict["variable_nodes"] = list(variable_nodes)

    # 总之到这为止
    # fixed_nodes包含的就是固定的node
    # variable_nodes包含的就是值会变化的node
    # 并且我们保证truth_value必定在variable nodes中

    # now page_data has the `doc_strings` attributes
    # and each field has the `is_truth_value_list` attributes

    # all_data_dict has the following attributes
    # "0000" ~ "1999" is the infomation for each page
    # "fixed_nodes" are the xpaths for nodes that cannot have truth-value
    # "variable_nodes" are the xpaths for nodes that might have truth-value

    return


def assure_value_variable(all_data_dict, variable_nodes, fixed_nodes, n_pages):
    """Makes sure all values are in the variable nodes by updating sets.

    Args:
      all_data_dict: the dictionary saving all data with groundtruth.
      variable_nodes: the current set of variable nodes.
      fixed_nodes: the current set of fixed nodes.
      n_pages: to assume we only process first n_pages pages from each website.
    """
    for index in all_data_dict:
        if not index.isdigit() or int(index) >= n_pages:
            # the key should be an integer, to exclude "fixed/variable nodes" entries.
            # n_pages to stop for only process part of the website.
            continue
        for field in all_data_dict[index]:
            if not field.startswith("field-"):
                continue
            xpaths = all_data_dict[index][field]["groundtruth_xpaths"]
            if not xpaths:  # There are zero value for this field in this page.
                continue
            flag = False
            for xpath in xpaths:
                if flag:  # The value's xpath is in the variable_nodes.
                    break
                flag = xpath in variable_nodes
            variable_nodes.update(xpaths)  # Add new xpaths if they are not in.
            fixed_nodes.difference_update(xpaths)


def generate_nodes_seq_and_write_to_file(compressed_args):
    """Extracts all the xpaths and labels the nodes for all the pages."""

    vertical, website = compressed_args

    all_data_dict = load_html_and_groundtruth(vertical, website)
    get_field_xpaths(
        all_data_dict,
        vertical_to_process=vertical,
        website_to_process=website,
        n_pages=2000,
        max_variable_nodes_per_website=300
    )
    """
    keys to the following example --->
        example = all_data_dict["0000"]

    dict_keys([
        'field-publication_date', 
        'field-author', 
        'field-title', 
        'field-publisher', 
        'field-isbn_13', 
        'html_str', 
        'path', 
        'dom_tree', 
        'xpath_data'
        ])
    """

    variable_nodes = all_data_dict["variable_nodes"]

    cleaned_features_for_this_website = {}

    for index in all_data_dict:
        if not index.isdigit():
            # Skip the cases when index is actually the "fixed/variable_nodes" keys.
            continue
        if int(index) >= FLAGS.n_pages:
            break
        page_data = all_data_dict[index]
        assert "xpath_data" in page_data

        doc_strings = page_data["doc_strings"]

        new_doc_strings = []

        field_info = {}
        for field in page_data:
            if not field.startswith("field-"):
                continue
            for doc_string_id in page_data[field]["is_truth_value_list"]:
                field_info[doc_string_id] = field[6:]

        for id, doc_string in enumerate(doc_strings):
            text, xpath = doc_string
            is_variable = xpath in variable_nodes
            if not is_variable:
                new_doc_strings.append((text, xpath, "fixed-node"))
            else:
                # for variable nodes,we need to give them labels
                gt_field = field_info.get(id, "none")
                new_doc_strings.append((text, xpath, gt_field))

        cleaned_features_for_this_website[index] = new_doc_strings

    output_file_path = os.path.join(FLAGS.output_data_path, f"{vertical}-{website}-{FLAGS.n_pages}.pickle")
    print(f"Writing the processed first {FLAGS.n_pages} pages of {vertical}-{website} into {output_file_path}")
    with open(output_file_path, "wb") as f:
        pickle.dump(cleaned_features_for_this_website, f)


def main(_):
    if not os.path.exists(FLAGS.output_data_path):
        os.makedirs(FLAGS.output_data_path)

    args_list = []

    vertical_to_websites_map = constants.VERTICAL_WEBSITES
    verticals = vertical_to_websites_map.keys()
    for vertical in verticals:
        websites = vertical_to_websites_map[vertical]
        for website in websites:
            args_list.append((vertical, website))

    num_cores = int(mp.cpu_count()/2)

    with mp.Pool(num_cores) as pool, tqdm(total=len(args_list), desc="Processing swde-data") as t:
        for res in pool.imap_unordered(generate_nodes_seq_and_write_to_file, args_list):
            t.update()


if __name__ == "__main__":
    app.run(main)
