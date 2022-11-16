import os
import torch
import collections
import logging
from tqdm import tqdm, trange
import json
import bs4
from os import path as osp
from bs4 import BeautifulSoup as bs
# from transformers.models.bert.tokenization_bert import BasicTokenizer, whitespace_tokenize
from torch.utils.data import Dataset
import networkx as nx
from lxml import etree
import pickle
# from transformers.tokenization_bert import BertTokenizer
from transformers import BertTokenizer
import argparse

tags_dict = {'a': 0, 'abbr': 1, 'acronym': 2, 'address': 3, 'altGlyph': 4, 'altGlyphDef': 5, 'altGlyphItem': 6,
             'animate': 7, 'animateColor': 8, 'animateMotion': 9, 'animateTransform': 10, 'applet': 11, 'area': 12,
             'article': 13, 'aside': 14, 'audio': 15, 'b': 16, 'base': 17, 'basefont': 18, 'bdi': 19, 'bdo': 20,
             'bgsound': 21, 'big': 22, 'blink': 23, 'blockquote': 24, 'body': 25, 'br': 26, 'button': 27, 'canvas': 28,
             'caption': 29, 'center': 30, 'circle': 31, 'cite': 32, 'clipPath': 33, 'code': 34, 'col': 35,
             'colgroup': 36, 'color-profile': 37, 'content': 38, 'cursor': 39, 'data': 40, 'datalist': 41, 'dd': 42,
             'defs': 43, 'del': 44, 'desc': 45, 'details': 46, 'dfn': 47, 'dialog': 48, 'dir': 49, 'div': 50, 'dl': 51,
             'dt': 52, 'ellipse': 53, 'em': 54, 'embed': 55, 'feBlend': 56, 'feColorMatrix': 57,
             'feComponentTransfer': 58, 'feComposite': 59, 'feConvolveMatrix': 60, 'feDiffuseLighting': 61,
             'feDisplacementMap': 62, 'feDistantLight': 63, 'feFlood': 64, 'feFuncA': 65, 'feFuncB': 66, 'feFuncG': 67,
             'feFuncR': 68, 'feGaussianBlur': 69, 'feImage': 70, 'feMerge': 71, 'feMergeNode': 72, 'feMorphology': 73,
             'feOffset': 74, 'fePointLight': 75, 'feSpecularLighting': 76, 'feSpotLight': 77, 'feTile': 78,
             'feTurbulence': 79, 'fieldset': 80, 'figcaption': 81, 'figure': 82, 'filter': 83, 'font-face-format': 84,
             'font-face-name': 85, 'font-face-src': 86, 'font-face-uri': 87, 'font-face': 88, 'font': 89, 'footer': 90,
             'foreignObject': 91, 'form': 92, 'frame': 93, 'frameset': 94, 'g': 95, 'glyph': 96, 'glyphRef': 97,
             'h1': 98, 'h2': 99, 'h3': 100, 'h4': 101, 'h5': 102, 'h6': 103, 'head': 104, 'header': 105, 'hgroup': 106,
             'hkern': 107, 'hr': 108, 'html': 109, 'i': 110, 'iframe': 111, 'image': 112, 'img': 113, 'input': 114,
             'ins': 115, 'kbd': 116, 'keygen': 117, 'label': 118, 'legend': 119, 'li': 120, 'line': 121,
             'linearGradient': 122, 'link': 123, 'main': 124, 'map': 125, 'mark': 126, 'marker': 127, 'marquee': 128,
             'mask': 129, 'math': 130, 'menu': 131, 'menuitem': 132, 'meta': 133, 'metadata': 134, 'meter': 135,
             'missing-glyph': 136, 'mpath': 137, 'nav': 138, 'nobr': 139, 'noembed': 140, 'noframes': 141,
             'noscript': 142, 'object': 143, 'ol': 144, 'optgroup': 145, 'option': 146, 'output': 147, 'p': 148,
             'param': 149, 'path': 150, 'pattern': 151, 'picture': 152, 'plaintext': 153, 'polygon': 154,
             'polyline': 155, 'portal': 156, 'pre': 157, 'progress': 158, 'q': 159, 'radialGradient': 160, 'rb': 161,
             'rect': 162, 'rp': 163, 'rt': 164, 'rtc': 165, 'ruby': 166, 's': 167, 'samp': 168, 'script': 169,
             'section': 170, 'select': 171, 'set': 172, 'shadow': 173, 'slot': 174, 'small': 175, 'source': 176,
             'spacer': 177, 'span': 178, 'stop': 179, 'strike': 180, 'strong': 181, 'style': 182, 'sub': 183,
             'summary': 184, 'sup': 185, 'svg': 186, 'switch': 187, 'symbol': 188, 'table': 189, 'tbody': 190,
             'td': 191, 'template': 192, 'text': 193, 'textPath': 194, 'textarea': 195, 'tfoot': 196, 'th': 197,
             'thead': 198, 'time': 199, 'title': 200, 'tr': 201, 'track': 202, 'tref': 203, 'tspan': 204, 'tt': 205,
             'u': 206, 'ul': 207, 'use': 208, 'var': 209, 'video': 210, 'view': 211, 'vkern': 212, 'wbr': 213,
             'xmp': 214}


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


# ---------- copied ! --------------
def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join([w for w in doc_tokens[new_start:(new_end + 1)]
                                  if w[0] != '<' or w[-1] != '>'])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end

class StrucDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (*torch.Tensor): tensors that have the same size of the first dimension.
        page_ids (list): the corresponding page ids of the input features.
        cnn_feature_dir (str): the direction where the cnn features are stored.
        token_to_tag (torch.Tensor): the mapping from each token to its corresponding tag id.
    """

    def __init__(self, *tensors, pad_id=0,
                 all_expended_attention_mask=None,
                 all_graph_names=None,
                 all_token_to_tag=None,
                 page_ids=None,
                 attention_width=None,
                 has_tree_attention_bias = False):
        tensors = tuple(tensor for tensor in tensors)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        if all_expended_attention_mask is not None:
            assert len(tensors[0]) == len(all_expended_attention_mask)
            tensors += (all_expended_attention_mask,)
        self.tensors = tensors
        self.page_ids = page_ids
        self.all_graph_names = all_graph_names
        self.all_token_to_tag = all_token_to_tag
        self.pad_id = pad_id
        self.attention_width = attention_width
        self.has_tree_attention_bias = has_tree_attention_bias

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]

        input_id = output[0]
        attention_mask = output[1]


        if not self.attention_width is None or self.has_tree_attention_bias:
            assert self.all_graph_names is not None , ("For non-empty attention_width / tree rel pos,"
                                                       "Graph names must be sent in!")

        if self.all_graph_names is not None:
            assert self.all_token_to_tag is not None
            graph_name = self.all_graph_names[index]
            token_to_tag = self.all_token_to_tag[index]
            with open(graph_name,"rb") as f:
                node_pairs_lengths = pickle.load(f)

            # node_pairs_lengths = dict(nx.all_pairs_shortest_path_length(graph))

            seq_len = len(token_to_tag)
            if self.has_tree_attention_bias:
                mat = [[0]*seq_len]*seq_len
            else:
                mat = None

            if self.attention_width is not None:
                emask = attention_mask.expand(seq_len,seq_len)
            else:
                emask = None

            for nid in range(seq_len):
                if input_id[nid]==self.pad_id:
                    break
                for anid in range(nid+1,seq_len):
                    if input_id[anid]==self.pad_id:
                        break

                    x_tid4nid = token_to_tag[nid]
                    x_tid4anid = token_to_tag[anid]

                    if x_tid4nid==x_tid4anid:
                        continue

                    try:
                        xx = node_pairs_lengths[x_tid4nid]
                        # x_tid4nid in valid tid list, or == -1
                    except:
                        # x_tid4nid out of bound, like `question`, `sep` or `cls`
                        xx = node_pairs_lengths[-1]
                        x_tid4nid=-1

                    try:
                        dis = xx[x_tid4anid]
                        # x_tid4anid in valid tid list, or == -1
                    except:
                        # x_tid4nid out of bound, like `question`, `sep` or `cls`
                        dis = xx[-1]
                        x_tid4anid = -1

                    # xx = node_pairs_lengths.get(tid4nid,node_pairs_lengths[-1])
                    # dis = xx.get(tid4anid,xx[-1])

                    if self.has_tree_attention_bias:
                        if x_tid4nid<x_tid4anid:
                            mat[nid][anid]=dis
                            mat[anid][nid]=-dis
                        else:
                            mat[nid][anid] = -dis
                            mat[anid][nid] = dis

                    if self.attention_width is not None:
                        # [nid][anid] determines whether nid can see anid
                        if x_tid4nid==-1 or x_tid4anid==-1: # sep / cls / question / pad
                            continue

                        if dis>self.attention_width:
                            emask[nid][anid]=0
                            emask[anid][nid]=0


            if self.attention_width is not None:
                output.append(emask)

            if self.has_tree_attention_bias:
                t_mat = torch.tensor(mat,dtype=torch.long)
                output.append(t_mat)


        return tuple(item for item in output)

    def __len__(self):
        return len(self.tensors[0])


def get_xpath4tokens(html_fn: str, unique_tids: set):
    xpath_map = {}
    tree = etree.parse(html_fn, etree.HTMLParser())
    nodes = tree.xpath('//*')
    for node in nodes:
        tid = node.attrib.get("tid")
        if int(tid) in unique_tids:
            xpath_map[int(tid)] = tree.getpath(node)
    xpath_map[len(nodes)] = "/html"
    xpath_map[len(nodes) + 1] = "/html"
    return xpath_map


def get_xpath_and_treeid4tokens(html_code, unique_tids, max_depth):
    unknown_tag_id = len(tags_dict)
    pad_tag_id = unknown_tag_id + 1
    max_width = 1000
    width_pad_id = 1001

    pad_x_tag_seq = [pad_tag_id] * max_depth
    pad_x_subs_seq = [width_pad_id] * max_depth
    pad_x_box = [0,0,0,0]
    pad_tree_id_seq = [width_pad_id] * max_depth

    def xpath_soup(element):

        xpath_tags = []
        xpath_subscripts = []
        tree_index = []
        child = element if element.name else element.parent
        for parent in child.parents:  # type: bs4.element.Tag
            siblings = parent.find_all(child.name, recursive=False)
            para_siblings = parent.find_all(True, recursive=False)
            xpath_tags.append(child.name)
            xpath_subscripts.append(
                0 if 1 == len(siblings) else next(i for i, s in enumerate(siblings, 1) if s is child))

            tree_index.append(next(i for i, s in enumerate(para_siblings, 0) if s is child))
            child = parent
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        tree_index.reverse()
        return xpath_tags, xpath_subscripts, tree_index

    xpath_tag_map = {}
    xpath_subs_map = {}
    tree_id_map = {}

    for tid in unique_tids:
        element = html_code.find(attrs={'tid': tid})
        if element is None:
            xpath_tags = pad_x_tag_seq
            xpath_subscripts = pad_x_subs_seq
            tree_index = pad_tree_id_seq

            xpath_tag_map[tid] = xpath_tags
            xpath_subs_map[tid] = xpath_subscripts
            tree_id_map[tid] = tree_index
            continue

        xpath_tags, xpath_subscripts, tree_index = xpath_soup(element)

        assert len(xpath_tags) == len(xpath_subscripts)
        assert len(xpath_tags) == len(tree_index)

        if len(xpath_tags) > max_depth:
            xpath_tags = xpath_tags[-max_depth:]
            xpath_subscripts = xpath_subscripts[-max_depth:]
            # tree_index = tree_index[-max_depth:]

        xpath_tags = [tags_dict.get(name, unknown_tag_id) for name in xpath_tags]
        xpath_subscripts = [min(i, max_width) for i in xpath_subscripts]
        tree_index = [min(i, max_width) for i in tree_index]

        # we do not append them to max depth here

        xpath_tags += [pad_tag_id] * (max_depth - len(xpath_tags))
        xpath_subscripts += [width_pad_id] * (max_depth - len(xpath_subscripts))
        # tree_index += [width_pad_id] * (max_depth - len(tree_index))

        xpath_tag_map[tid] = xpath_tags
        xpath_subs_map[tid] = xpath_subscripts
        tree_id_map[tid] = tree_index

    return xpath_tag_map, xpath_subs_map, tree_id_map


# ---------- copied ! --------------
def _check_is_max_context(doc_spans, cur_span_index, position):
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class SRCExample(object):
    r"""
    The Containers for SRC Examples.

    Arguments:
        doc_tokens (list[str]): the original tokens of the HTML file before dividing into sub-tokens.
        qas_id (str): the id of the corresponding question.
        tag_num (int): the total tag number in the corresponding HTML file, including the additional 'yes' and 'no'.
        question_text (str): the text of the corresponding question.
        orig_answer_text (str): the answer text provided by the dataset.
        all_doc_tokens (list[str]): the sub-tokens of the corresponding HTML file.
        start_position (int): the position where the answer starts in the all_doc_tokens.
        end_position (int): the position where the answer ends in the all_doc_tokens; NOTE that the answer tokens
                            include the token at end_position.
        tok_to_orig_index (list[int]): the mapping from sub-tokens (all_doc_tokens) to origin tokens (doc_tokens).
        orig_to_tok_index (list[int]): the mapping from origin tokens (doc_tokens) to sub-tokens (all_doc_tokens).
        tok_to_tags_index (list[int]): the mapping from sub-tokens (all_doc_tokens) to the id of the deepest tag it
                                       belongs to.
    """

    # the difference between T-PLM and H-PLM is just add <xx> and </xx> into the
    # original tokens and further-tokenized tokens
    def __init__(self,
                 doc_tokens,
                 qas_id,
                 tag_num,  # <xx> ?? </xx> is counted as one tag
                 question_text=None,
                 html_code=None,
                 orig_answer_text=None,
                 start_position=None,  # in all_doc_tokens
                 end_position=None,  # in all_doc_tokens
                 tok_to_orig_index=None,
                 orig_to_tok_index=None,
                 all_doc_tokens=None,
                 tok_to_tags_index=None,
                 xpath_tag_map=None,
                 xpath_subs_map=None,
                 xpath_box=None,
                 tree_id_map=None,
                 visible_matrix=None,
                 ):
        self.doc_tokens = doc_tokens
        self.qas_id = qas_id
        self.tag_num = tag_num
        self.question_text = question_text
        self.html_code = html_code
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.tok_to_orig_index = tok_to_orig_index
        self.orig_to_tok_index = orig_to_tok_index
        self.all_doc_tokens = all_doc_tokens
        self.tok_to_tags_index = tok_to_tags_index
        self.xpath_tag_map = xpath_tag_map
        self.xpath_subs_map = xpath_subs_map
        self.xpath_box = xpath_box
        self.tree_id_map = tree_id_map
        self.visible_matrix = visible_matrix

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        """
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.end_position:
            s += ", end_position: %d" % self.end_position
        """
        s = "[INFO]\n"
        s += f"qas_id ({type(self.qas_id)}): {self.qas_id}\n"
        s += f"tag_num ({type(self.tag_num)}): {self.tag_num}\n"
        s += f"question_text ({type(self.question_text)}): {self.question_text}\n"
        s += f"html_code ({type(self.html_code)}): {self.html_code}\n"
        s += f"orig_answer_text ({type(self.orig_answer_text)}): {self.orig_answer_text}\n"
        s += f"start_position ({type(self.start_position)}): {self.start_position}\n"
        s += f"end_position ({type(self.end_position)}): {self.end_position}\n"
        s += f"tok_to_orig_index ({type(self.tok_to_orig_index)}): {self.tok_to_orig_index}\n"
        s += f"orig_to_tok_index ({type(self.orig_to_tok_index)}): {self.orig_to_tok_index}\n"
        s += f"all_doc_tokens ({type(self.all_doc_tokens)}): {self.all_doc_tokens}\n"
        s += f"tok_to_tags_index ({type(self.tok_to_tags_index)}): {self.tok_to_tags_index}\n"
        s += f"xpath_tag_map ({type(self.xpath_tag_map)}): {self.xpath_tag_map}\n"
        s += f"xpath_subs_map ({type(self.xpath_subs_map)}): {self.xpath_subs_map}\n"
        s += f"tree_id_map ({type(self.tree_id_map)}): {self.tree_id_map}\n"

        return s




class InputFeatures(object):
    r"""
    The Container for the Features of Input Doc Spans.

    Arguments:
        unique_id (int): the unique id of the input doc span.
        example_index (int): the index of the corresponding SRC Example of the input doc span.
        page_id (str): the id of the corresponding web page of the question.
        doc_span_index (int): the index of the doc span among all the doc spans which corresponding to the same SRC
                              Example.
        tokens (list[str]): the sub-tokens of the input sequence, including cls token, sep tokens, and the sub-tokens
                            of the question and HTML file.
        token_to_orig_map (dict[int, int]): the mapping from the HTML file's sub-tokens in the sequence tokens (tokens)
                                            to the origin tokens (all_tokens in the corresponding SRC Example).
        token_is_max_context (dict[int, bool]): whether the current doc span contains the max pre- and post-context for
                                                each HTML file's sub-tokens.
        input_ids (list[int]): the ids of the sub-tokens in the input sequence (tokens).
        input_mask (list[int]): use 0/1 to distinguish the input sequence from paddings.
        segment_ids (list[int]): use 0/1 to distinguish the question and the HTML files.
        paragraph_len (int): the length of the HTML file's sub-tokens.
        start_position (int): the position where the answer starts in the input sequence (0 if the answer is not fully
                              in the input sequence).
        end_position (int): the position where the answer ends in the input sequence; NOTE that the answer tokens
                            include the token at end_position (0 if the answer is not fully in the input sequence).
        token_to_tag_index (list[int]): the mapping from sub-tokens of the input sequence to the id of the deepest tag
                                        it belongs to.
        is_impossible (bool): whether the answer is fully in the doc span.
    """

    def __init__(self,
                 unique_id,
                 example_index,
                 page_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 token_to_tag_index=None,
                 is_impossible=None,
                 xpath_tags_seq=None,
                 xpath_subs_seq=None,
                 xpath_box_seq=None,
                 extended_attention_mask=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.page_id = page_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.token_to_tag_index = token_to_tag_index
        self.is_impossible = is_impossible
        self.xpath_tags_seq = xpath_tags_seq
        self.xpath_subs_seq = xpath_subs_seq
        self.xpath_box_seq = xpath_box_seq
        self.extended_attention_mask = extended_attention_mask


def html_escape(html):
    r"""
    replace the special expressions in the html file for specific punctuation.
    """
    html = html.replace('&quot;', '"')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&nbsp;', ' ')
    return html

def read_squad_examples(args, input_file, root_dir, is_training, tokenizer, simplify=False, max_depth=50,
                        split_flag="n-eon",
                        attention_width=None):
    r"""
    pre-process the data in json format into SRC Examples.

    Arguments:
        split_flag:
        attention_width:
        input_file (str): the inputting data file in json format.
        root_dir (str): the root directory of the raw WebSRC dataset, which contains the HTML files.
        is_training (bool): True if processing the training set, else False.
        tokenizer (Tokenizer): the tokenizer for PLM in use.
        method (str): the name of the method in use, choice: ['T-PLM', 'H-PLM', 'V-PLM'].
        simplify (bool): when setting to Ture, the returned Example will only contain document tokens, the id of the
                         question-answers, and the total tag number in the corresponding html files.
    Returns:
        list[SRCExamples]: the resulting SRC Examples, contained all the needed information for the feature generation
                           process, except when the argument simplify is setting to True;
        set[str]: all the tag names appeared in the processed dataset, e.g. <div>, <img/>, </p>, etc..
    """
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    pad_tree_id_seq = [1001] * max_depth

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def html_to_text_list(h):
        tag_num, text_list = 0, []
        for element in h.descendants:
            if (type(element) == bs4.element.NavigableString) and (element.strip()):
                text_list.append(element.strip())
            if type(element) == bs4.element.Tag:
                tag_num += 1
        return text_list, tag_num + 2  # + 2 because we treat the additional 'yes' and 'no' as two special tags.

    def html_to_text(h):
        tag_list = set()
        for element in h.descendants:
            if type(element) == bs4.element.Tag:
                element.attrs = {}
                temp = str(element).split()
                tag_list.add(temp[0])
                tag_list.add(temp[-1])
        return html_escape(str(h)), tag_list

    def adjust_offset(offset, text):
        text_list = text.split()
        cnt, adjustment = 0, []
        for t in text_list:
            if not t:
                continue
            if t[0] == '<' and t[-1] == '>':
                adjustment.append(offset.index(cnt))
            else:
                cnt += 1
        add = 0
        adjustment.append(len(offset))
        for i in range(len(offset)):
            while i >= adjustment[add]:
                add += 1
            offset[i] += add
        return offset

    def e_id_to_t_id(e_id, html):
        t_id = 0
        for element in html.descendants:
            if type(element) == bs4.element.NavigableString and element.strip():
                t_id += 1
            if type(element) == bs4.element.Tag:
                if int(element.attrs['tid']) == e_id:
                    break
        return t_id

    def calc_num_from_raw_text_list(t_id, l):
        n_char = 0
        for i in range(t_id):
            n_char += len(l[i]) + 1
        return n_char

    def word_to_tag_from_text(tokens, h):
        cnt, w_t, path = -1, [], []
        unique_tids = set()
        for t in tokens[0:-2]:
            if len(t) < 2:
                w_t.append(path[-1])
                unique_tids.add(path[-1])
                continue
            if t[0] == '<' and t[-2] == '/':
                cnt += 1
                w_t.append(cnt)
                unique_tids.add(cnt)
                continue
            if t[0] == '<' and t[1] != '/':
                cnt += 1
                path.append(cnt)
            w_t.append(path[-1])
            unique_tids.add(path[-1])
            if t[0] == '<' and t[1] == '/':
                del path[-1]
        w_t.append(cnt + 1)
        unique_tids.add(cnt + 1)
        w_t.append(cnt + 2)
        unique_tids.add(cnt + 2)
        assert len(w_t) == len(tokens)
        assert len(path) == 0, print(h)
        return w_t, unique_tids

    def word_tag_offset(html):
        cnt, w_t, t_w, tags, tags_tids = 0, [], [], [], []
        for element in html.descendants:
            if type(element) == bs4.element.Tag:
                content = ' '.join(list(element.strings)).split()
                t_w.append({'start': cnt, 'len': len(content)})
                tags.append('<' + element.name + '>')
                tags_tids.append(element['tid'])
            elif type(element) == bs4.element.NavigableString and element.strip():
                text = element.split()
                tid = element.parent['tid']
                ind = tags_tids.index(tid)
                for _ in text:
                    w_t.append(ind)
                    cnt += 1
                assert cnt == len(w_t)
        w_t.append(len(t_w))
        w_t.append(len(t_w) + 1)
        return w_t

    def subtoken_tag_offset(html, s_tok):
        w_t = word_tag_offset(html)
        s_t = []
        unique_tids = set()
        for i in range(len(s_tok)):
            s_t.append(w_t[s_tok[i]])
            unique_tids.add(w_t[s_tok[i]])
        return s_t, unique_tids

    def subtoken_tag_offset_plus_eon(html, s_tok, all_doc_tokens):
        w_t = word_tag_offset(html)
        s_t = []
        unique_tids = set()
        offset = 0
        for i in range(len(s_tok)):
            if all_doc_tokens[i] not in ('<end-of-node>', tokenizer.sep_token, tokenizer.cls_token):
                s_t.append(w_t[s_tok[i] - offset])
                unique_tids.add(w_t[s_tok[i] - offset])
            else:
                prev_tid = s_t[-1]
                s_t.append(prev_tid)
                offset += 1
        return s_t, unique_tids

    def check_visible(path1, path2, attention_width):
        i = 0
        j = 0
        dis = 0
        lp1 = len(path1)
        lp2 = len(path2)
        while i < lp1 and j < lp2 and path1[i] == path2[j]:
            i += 1
            j += 1

        if i < lp1 and j < lp2:
            dis += lp1 - i + lp2 - j
        else:
            if i == lp1:
                dis += lp2 - j
            else:
                dis += lp1 - i

        if dis <= attention_width:
            return True
        return False


    def from_tids_to_box(html_fn, unique_tids, json_fn):
        sorted_ids = sorted(unique_tids)
        f = open(json_fn, 'r')
        data = json.load(f)
        orig_width, orig_height = data['2']['rect']['width'], data['2']['rect']['height']
        orig_x, orig_y = data['2']['rect']['x'], data['2']['rect']['y']

        return_dict = {}
        for id in sorted_ids:
            if str(id) in data:
                x, y, width, height = data[str(id)]['rect']['x'], data[str(id)]['rect']['y'], data[str(id)]['rect']['width'], data[str(id)]['rect']['height']
                resize_x = (x - orig_x) * 1000 // orig_width
                resize_y = (y - orig_y) * 1000 // orig_height
                
                resize_width = width * 1000 // orig_width
                resize_height = height * 1000 // orig_height

                # if not (resize_x <= 1000 and resize_y <= 1000):
                #     print('before', x, y, width, height)
                #     print('after', resize_x, resize_y, resize_width, resize_height)
                #     print('file name ', html_fn)
                #     # exit(0)

                if resize_x < 0 or resize_y < 0 or resize_width < 0 or resize_height < 0: # meaningless
                    return_dict[id] = [0, 0, 0, 0]
                else:
                    return_dict[id] = [int(resize_x), int(resize_y), int(resize_x+resize_width), int(resize_y+resize_height)]
            else:
                return_dict[id] = [0,0,0,0]
       
        return return_dict

    def get_visible_matrix(unique_tids, tree_id_map, attention_width):
        if attention_width is None:
            return None
        unique_tids_list = list(unique_tids)
        visible_matrix = collections.defaultdict(list)
        for i in range(len(unique_tids_list)):
            if tree_id_map[unique_tids_list[i]] == pad_tree_id_seq:
                visible_matrix[unique_tids_list[i]] = list()
                continue
            visible_matrix[unique_tids_list[i]].append(unique_tids_list[i])
            for j in range(i + 1, len(unique_tids_list)):
                if check_visible(tree_id_map[unique_tids_list[i]], tree_id_map[unique_tids_list[j]], attention_width):
                    visible_matrix[unique_tids_list[i]].append(unique_tids_list[j])
                    visible_matrix[unique_tids_list[j]].append(unique_tids_list[i])
        return visible_matrix

    examples = []
    all_tag_list = set()
    total_num = sum([len(entry["websites"]) for entry in input_data])
    with tqdm(total=total_num, desc="Converting websites to examples") as t:
        for entry in input_data:
            # print('entry', entry)
            domain = entry["domain"]
            for website in entry["websites"]:
                # print('website', website)

                # Generate Doc Tokens
                page_id = website["page_id"]
                # print('page_id', page_id)
                curr_dir = osp.join(root_dir, domain, page_id[0:2], 'processed_data')
                html_fn = osp.join(curr_dir, page_id + '.html')
                json_fn = osp.join(curr_dir, page_id + '.json')

                # print('html', html_fn)

                html_file = open(html_fn).read()
                html_code = bs(html_file, "html.parser")
                raw_text_list, tag_num = html_to_text_list(html_code) 

                # print(raw_text_list)
                # print(tag_num)
                # exit(0)
                doc_tokens = []
                char_to_word_offset = []

                # print(split_flag) # n-eon
                # exit(0)

                if split_flag in ["y-eon", "y-sep", "y-cls"]:
                    prev_is_whitespace = True
                    for i, doc_string in enumerate(raw_text_list):
                        for c in doc_string:
                            if is_whitespace(c):
                                prev_is_whitespace = True
                            else:
                                if prev_is_whitespace:
                                    doc_tokens.append(c)
                                else:
                                    doc_tokens[-1] += c
                                prev_is_whitespace = False
                            char_to_word_offset.append(len(doc_tokens) - 1)

                        if i < len(raw_text_list) - 1:
                            prev_is_whitespace = True
                            char_to_word_offset.append(len(doc_tokens) - 1)

                        if split_flag == "y-eon":
                            doc_tokens.append('<end-of-node>')
                        elif split_flag == "y-sep":
                            doc_tokens.append(tokenizer.sep_token)
                        elif split_flag == "y-cls":
                            doc_tokens.append(tokenizer.cls_token)
                        else:
                            raise ValueError("Split flag should be `y-eon` or `y-sep` or `y-cls`")
                        prev_is_whitespace = True

                elif split_flag =="n-eon" or split_flag == "y-hplm":
                    page_text = ' '.join(raw_text_list) 
                    prev_is_whitespace = True
                    for c in page_text:
                        if is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1) 


                doc_tokens.append('no')
                char_to_word_offset.append(len(doc_tokens) - 1)
                doc_tokens.append('yes')
                char_to_word_offset.append(len(doc_tokens) - 1)

                if split_flag == "y-hplm":
                    real_text, tag_list = html_to_text(bs(html_file))
                    all_tag_list = all_tag_list | tag_list
                    char_to_word_offset = adjust_offset(char_to_word_offset, real_text)
                    doc_tokens = real_text.split()
                    doc_tokens.append('no')
                    doc_tokens.append('yes')
                    doc_tokens = [i for i in doc_tokens if i]

                else:
                    tag_list = []

                assert len(doc_tokens) == char_to_word_offset[-1] + 1, (len(doc_tokens), char_to_word_offset[-1])

                if simplify:
                    for qa in website["qas"]:
                        qas_id = qa["id"]
                        example = SRCExample(doc_tokens=doc_tokens, qas_id=qas_id, tag_num=tag_num)
                        examples.append(example)
                    t.update(1)
                else:
                    # Tokenize all doc tokens
                    # tokenize sth like < / >
                    tok_to_orig_index = []
                    orig_to_tok_index = []
                    all_doc_tokens = []
                    for (i, token) in enumerate(doc_tokens):
                        orig_to_tok_index.append(len(all_doc_tokens))
                        if token in tag_list:
                            sub_tokens = [token]
                        else:
                            sub_tokens = tokenizer.tokenize(token)
                        for sub_token in sub_tokens:
                            tok_to_orig_index.append(i)
                            all_doc_tokens.append(sub_token)

                    # Generate extra information for features
                    if split_flag in ["y-eon", "y-sep", "y-cls"]:
                        tok_to_tags_index, unique_tids = subtoken_tag_offset_plus_eon(html_code, tok_to_orig_index,
                                                                                      all_doc_tokens)
                    elif split_flag == "n-eon":
                        tok_to_tags_index, unique_tids = subtoken_tag_offset(html_code, tok_to_orig_index)

                    elif split_flag == "y-hplm":
                        tok_to_tags_index, unique_tids = word_to_tag_from_text(all_doc_tokens, html_code)

                    else:
                        raise ValueError("Unsupported split_flag!")

                

                    xpath_tag_map, xpath_subs_map, tree_id_map = get_xpath_and_treeid4tokens(html_code, unique_tids,
                                                                                             max_depth=max_depth)
                    # tree_id_map : neither truncated nor padded
                    xpath_box = from_tids_to_box(html_fn, unique_tids, json_fn) 


                    assert tok_to_tags_index[-1] == tag_num - 1, (tok_to_tags_index[-1], tag_num - 1)

                    # we get attention_mask here
                    visible_matrix = get_visible_matrix(unique_tids, tree_id_map, attention_width=attention_width)

                    # Process each qas, which is mainly calculate the answer position
                    for qa in website["qas"]:
                        qas_id = qa["id"]
                        question_text = qa["question"]
                        start_position = None
                        end_position = None
                        orig_answer_text = None

                        if is_training:
                            if len(qa["answers"]) != 1:
                                raise ValueError(
                                    "For training, each question should have exactly 1 answer.")
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            if answer["element_id"] == -1:
                                num_char = len(char_to_word_offset) - 2
                            else:
                                num_char = calc_num_from_raw_text_list(e_id_to_t_id(answer["element_id"], html_code),
                                                                       raw_text_list)
                            answer_offset = num_char + answer["answer_start"]
                            answer_length = len(orig_answer_text) if answer["element_id"] != -1 else 1
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            actual_text = " ".join([w for w in doc_tokens[start_position:(end_position + 1)]
                                                    if (w[0] != '<' or w[-1] != '>')
                                                    and w != "<end-of-node>"
                                                    and w != tokenizer.sep_token
                                                    and w != tokenizer.cls_token])
                            cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logging.warning("Could not find answer of question %s: '%s' vs. '%s'",
                                               qa['id'], actual_text, cleaned_answer_text)
                                continue

                        example = SRCExample(
                            doc_tokens=doc_tokens,
                            qas_id=qas_id,
                            tag_num=tag_num,
                            question_text=question_text,
                            html_code=html_code,
                            orig_answer_text=orig_answer_text,
                            start_position=start_position,
                            end_position=end_position,
                            tok_to_orig_index=tok_to_orig_index,
                            orig_to_tok_index=orig_to_tok_index,
                            all_doc_tokens=all_doc_tokens,
                            tok_to_tags_index=tok_to_tags_index,
                            xpath_tag_map=xpath_tag_map,
                            xpath_subs_map=xpath_subs_map,
                            xpath_box=xpath_box, 
                            tree_id_map=tree_id_map,
                            visible_matrix=visible_matrix
                        )

                        examples.append(example)

                        
                        if args.web_num_features != 0:
                            if len(examples) >= args.web_num_features:
                                return examples, all_tag_list

                    t.update(1)
    return examples, all_tag_list


def load_and_cache_examples(args, tokenizer, max_depth=50, evaluate=False, output_examples=False):
    r"""
    Load and process the raw data.
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.web_eval_file if evaluate else args.web_train_file

    cached_features_file = os.path.join(args.cache_dir, 'cached_{}_{}_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        "markuplm",
        str(args.max_seq_length),
        str(max_depth),
        args.web_num_features,
        args.model_type
    ))
    if not os.path.exists(os.path.dirname(cached_features_file)):
        os.makedirs(os.path.dirname(cached_features_file))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        if output_examples:
            examples, tag_list = read_squad_examples(args, input_file=input_file,
                                                     root_dir=args.web_root_dir,
                                                     is_training=not evaluate,
                                                     tokenizer=tokenizer,
                                                     simplify=True,
                                                     max_depth=max_depth
                                                     )
        else:
            examples = None
    else:
        print("Creating features from dataset file at %s", input_file)

        examples, _ = read_squad_examples(args, input_file=input_file,
                                          root_dir=args.web_root_dir,
                                          is_training=not evaluate,
                                          tokenizer=tokenizer,
                                          simplify=False,
                                          max_depth=max_depth)

        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate,
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                pad_token=tokenizer.pad_token_id,
                                                sequence_a_segment_id=0,
                                                sequence_b_segment_id=0,
                                                max_depth=max_depth)

        if args.local_rank in [-1, 0] and args.web_save_features:
            print("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    all_xpath_tags_seq = torch.tensor([f.xpath_tags_seq for f in features], dtype=torch.long)
    all_xpath_subs_seq = torch.tensor([f.xpath_subs_seq for f in features], dtype=torch.long)
    all_xpath_box_seq = torch.tensor([f.xpath_box_seq for f in features], dtype=torch.long)


    if evaluate:
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_index,
                               all_xpath_tags_seq, all_xpath_subs_seq, all_xpath_box_seq)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_xpath_tags_seq, all_xpath_subs_seq,
                               all_start_positions, all_end_positions, all_xpath_box_seq)

    if output_examples:
        dataset = (dataset, examples, features)
    return dataset




def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, max_depth=50):
    r"""
    Converting the SRC Examples further into the features for all the input doc spans.

    Arguments:
        examples (list[SRCExample]): the list of SRC Examples to process.
        tokenizer (Tokenizer): the tokenizer for PLM in use.
        max_seq_length (int): the max length of the total sub-token sequence, including the question, cls token, sep
                              tokens, and documents; if the length of the input is bigger than max_seq_length, the input
                              will be cut into several doc spans.
        doc_stride (int): the stride length when the input is cut into several doc spans.
        max_query_length (int): the max length of the sub-token sequence of the questions; the question will be truncate
                                if it is longer than max_query_length.
        is_training (bool): True if processing the training set, else False.
        cls_token (str): the cls token in use, default is '[CLS]'.
        sep_token (str): the sep token in use, default is '[SEP]'.
        pad_token (int): the id of the padding token in use when the total sub-token length is smaller that
                         max_seq_length, default is 0 which corresponding to the '[PAD]' token.
        sequence_a_segment_id: the segment id for the first sequence (the question), default is 0.
        sequence_b_segment_id: the segment id for the second sequence (the html file), default is 1.
        cls_token_segment_id: the segment id for the cls token, default is 0.
        pad_token_segment_id: the segment id for the padding tokens, default is 0.
        mask_padding_with_zero: determine the pattern of the returned input mask; 0 for padding tokens and 1 for others
                                when True, and vice versa.
    Returns:
        list[InputFeatures]: the resulting input features for all the input doc spans
    """

    pad_x_tag_seq = [216] * max_depth
    pad_x_subs_seq = [1001] * max_depth
    pad_x_box = [0,0,0,0]
    pad_tree_id_seq = [1001] * max_depth

    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(tqdm(examples, desc="Converting examples to features")):

        xpath_tag_map = example.xpath_tag_map
        xpath_subs_map = example.xpath_subs_map
        xpath_box = example.xpath_box  
        tree_id_map = example.tree_id_map
        visible_matrix = example.visible_matrix

        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_start_position = None
        tok_end_position = None
        if is_training:
            tok_start_position = example.orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = example.orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(example.all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                example.all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(example.all_doc_tokens):
            length = len(example.all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(example.all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            token_to_tag_index = []

            # CLS token at the beginning
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            token_to_tag_index.append(example.tag_num)

            # Query
            tokens += query_tokens
            segment_ids += [sequence_a_segment_id] * len(query_tokens)
            token_to_tag_index += [example.tag_num] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            token_to_tag_index.append(example.tag_num)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = example.tok_to_orig_index[split_token_index]
                token_to_tag_index.append(example.tok_to_tags_index[split_token_index])

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(example.all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            token_to_tag_index.append(example.tag_num)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                token_to_tag_index.append(example.tag_num)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(token_to_tag_index) == max_seq_length

            span_is_impossible = False
            start_position = None
            end_position = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    span_is_impossible = True
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            '''
            if 10 < example_index < 20:
                print("*** Example ***")
                #print("page_id: %s" % (example.qas_id[:-5]))
                #print("token_to_tag_index  :%s" % token_to_tag_index)
                #print(len(token_to_tag_index))
                #print("unique_id: %s" % (unique_id))
                #print("example_index: %s" % (example_index))
                #print("doc_span_index: %s" % (doc_span_index))
                # print("tokens: %s" % " ".join(tokens))

                print("tokens: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in enumerate(tokens)
                ]))

                #print("token_to_orig_map: %s" % " ".join([
                #    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                #print(len(token_to_orig_map))
                # print("token_is_max_context: %s" % " ".join([
                #    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                # ]))
                #print(len(token_is_max_context))
                #print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                #print(len(input_ids))
                #print(
                #    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                #print(len(input_mask))
                #print(
                #    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                #print(len(segment_ids))
                print(f"original answer: {example.orig_answer_text}")
                if is_training and span_is_impossible:
                    print("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    print("start_position: %d" % (start_position))
                    print("end_position: %d" % (end_position))
                    print(
                        "answer: %s" % (answer_text))
            '''

            # print('token_to_tag_index', token_to_tag_index)
            # print('xpath_tag_map', xpath_tag_map)
            # exit(0)

            xpath_tags_seq = [xpath_tag_map.get(tid, pad_x_tag_seq) for tid in token_to_tag_index]  # ok
            xpath_subs_seq = [xpath_subs_map.get(tid, pad_x_subs_seq) for tid in token_to_tag_index]  # ok
            xpath_box_seq = [xpath_box.get(tid, pad_x_box) for tid in token_to_tag_index]
            # print(xpath_box_seq)
            # exit(0)

            # we need to get extended_attention_mask
            if visible_matrix is not None:
                extended_attention_mask = []
                for tid in token_to_tag_index:
                    if tid == example.tag_num:
                        extended_attention_mask.append(input_mask)
                    else:
                        visible_tids = visible_matrix[tid]
                        if len(visible_tids) == 0:
                            extended_attention_mask.append(input_mask)
                            continue
                        visible_per_token = []
                        for i, tid in enumerate(token_to_tag_index):
                            if tid == example.tag_num and input_mask[i] == (1 if mask_padding_with_zero else 0):
                                visible_per_token.append(1 if mask_padding_with_zero else 0)
                            elif tid in visible_tids:
                                visible_per_token.append(1 if mask_padding_with_zero else 0)
                            else:
                                visible_per_token.append(0 if mask_padding_with_zero else 1)
                        extended_attention_mask.append(visible_per_token)  # should be (max_seq_len*max_seq_len)
            else:
                extended_attention_mask = None

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    page_id=example.qas_id[:-5],
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    token_to_tag_index=token_to_tag_index,
                    is_impossible=span_is_impossible,
                    xpath_tags_seq=xpath_tags_seq,
                    xpath_subs_seq=xpath_subs_seq,
                    xpath_box_seq=xpath_box_seq,
                    extended_attention_mask=extended_attention_mask,
                ))
            unique_id += 1

    return features

def get_websrc_dataset(args, tokenizer, evaluate=False, output_examples=False):
    if not evaluate:
        websrc_dataset = load_and_cache_examples(args, tokenizer, evaluate=evaluate, output_examples=False)
        return websrc_dataset
    else:
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=evaluate, output_examples=True)
        return dataset, examples, features
