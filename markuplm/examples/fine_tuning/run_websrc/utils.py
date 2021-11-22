from __future__ import absolute_import, division, print_function
import json
import logging
import math
import collections
from io import open
from os import path as osp

from tqdm import tqdm
import bs4
from bs4 import BeautifulSoup as bs
from transformers.models.bert.tokenization_bert import BasicTokenizer, whitespace_tokenize
from torch.utils.data import Dataset
from lxml import etree
from markuplmft.data.tag_utils import tags_dict

logger = logging.getLogger(__name__)


class StrucDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (*torch.Tensor): tensors that have the same size of the first dimension.
        page_ids (list): the corresponding page ids of the input features.
        cnn_feature_dir (str): the direction where the cnn features are stored.
        token_to_tag (torch.Tensor): the mapping from each token to its corresponding tag id.
    """

    def __init__(self, *tensors):
        tensors = tuple(tensor for tensor in tensors)
        assert all(len(tensors[0]) == len(tensor) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        output = [tensor[index] for tensor in self.tensors]
        return tuple(item for item in output)

    def __len__(self):
        return len(self.tensors[0])


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
                 xpath_subs_seq=None
                 ):
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

    for tid in unique_tids:
        element = html_code.find(attrs={'tid': tid})
        if element is None:
            xpath_tags = pad_x_tag_seq
            xpath_subscripts = pad_x_subs_seq

            xpath_tag_map[tid] = xpath_tags
            xpath_subs_map[tid] = xpath_subscripts
            continue

        xpath_tags, xpath_subscripts, tree_index = xpath_soup(element)

        assert len(xpath_tags) == len(xpath_subscripts)
        assert len(xpath_tags) == len(tree_index)

        if len(xpath_tags) > max_depth:
            xpath_tags = xpath_tags[-max_depth:]
            xpath_subscripts = xpath_subscripts[-max_depth:]

        xpath_tags = [tags_dict.get(name, unknown_tag_id) for name in xpath_tags]
        xpath_subscripts = [min(i, max_width) for i in xpath_subscripts]

        # we do not append them to max depth here

        xpath_tags += [pad_tag_id] * (max_depth - len(xpath_tags))
        xpath_subscripts += [width_pad_id] * (max_depth - len(xpath_subscripts))

        xpath_tag_map[tid] = xpath_tags
        xpath_subs_map[tid] = xpath_subscripts

    return xpath_tag_map, xpath_subs_map


def read_squad_examples(input_file, root_dir, is_training, tokenizer, simplify=False, max_depth=50):
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

    examples = []
    all_tag_list = set()
    total_num = sum([len(entry["websites"]) for entry in input_data])
    with tqdm(total=total_num, desc="Converting websites to examples") as t:
        for entry in input_data:
            domain = entry["domain"]
            for website in entry["websites"]:

                # Generate Doc Tokens
                page_id = website["page_id"]
                curr_dir = osp.join(root_dir, domain, page_id[0:2], 'processed_data')
                html_fn = osp.join(curr_dir, page_id + '.html')

                html_file = open(html_fn).read()
                html_code = bs(html_file, "html.parser")
                raw_text_list, tag_num = html_to_text_list(html_code)  # 字符列表及标签数

                doc_tokens = []
                char_to_word_offset = []

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
                    tok_to_tags_index, unique_tids = subtoken_tag_offset(html_code, tok_to_orig_index)

                    xpath_tag_map, xpath_subs_map = get_xpath_and_treeid4tokens(html_code,
                                                                                unique_tids,
                                                                                max_depth=max_depth)

                    assert tok_to_tags_index[-1] == tag_num - 1, (tok_to_tags_index[-1], tag_num - 1)

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
                                                    if (w[0] != '<' or w[-1] != '>')])
                            cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("Could not find answer of question %s: '%s' vs. '%s'",
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
                        )

                        examples.append(example)

                    t.update(1)
    return examples, all_tag_list


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

    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(tqdm(examples, desc="Converting examples to features")):

        xpath_tag_map = example.xpath_tag_map
        xpath_subs_map = example.xpath_subs_map

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


            xpath_tags_seq = [xpath_tag_map.get(tid, pad_x_tag_seq) for tid in token_to_tag_index]  # ok
            xpath_subs_seq = [xpath_subs_map.get(tid, pad_x_subs_seq) for tid in token_to_tag_index]  # ok

            # we need to get extended_attention_mask

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
                ))
            unique_id += 1

    return features


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


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size, max_answer_length, do_lower_case,
                      output_prediction_file, output_tag_prediction_file,
                      output_nbest_file, verbose_logging, tokenizer):
    r"""
    Compute and write down the final results, including the n best results.

    Arguments:
        all_examples (list[SRCExample]): all the SRC Example of the dataset; note that we only need it to provide the
                                         mapping from example index to the question-answers id.
        all_features (list[InputFeatures]): all the features for the input doc spans.
        all_results (list[RawResult]): all the results from the models.
        n_best_size (int): the number of the n best buffer and the final n best result saved.
        max_answer_length (int): constrain the model to predict the answer no longer than it.
        do_lower_case (bool): whether the model distinguish upper and lower case of the letters.
        output_prediction_file (str): the file which the best answer text predictions will be written to.
        output_tag_prediction_file (str): the file which the best answer tag predictions will be written to.
        output_nbest_file (str): the file which the n best answer predictions including text, tag, and probabilities
                                 will be written to.
        verbose_logging (bool): if true, all of the warnings related to data processing will be printed.
    """
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "tag_ids"])

    all_predictions = collections.OrderedDict()
    all_tag_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []

        for (feature_index, feature) in enumerate(features):

            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    tag_ids = set(feature.token_to_tag_index[start_index: end_index + 1])
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            tag_ids=list(tag_ids)))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "tag_ids"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = _get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    tag_ids=pred.tag_ids))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, tag_ids=[-1]))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["tag_ids"] = entry.tag_ids
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        best = nbest_json[0]["text"].split()
        best = ' '.join([w for w in best
                         if (w[0] != '<' or w[-1] != '>')
                         and w != "<end-of-node>"
                         and w != tokenizer.sep_token
                         and w != tokenizer.cls_token])
        all_predictions[example.qas_id] = best
        all_tag_predictions[example.qas_id] = nbest_json[0]["tag_ids"]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    with open(output_tag_prediction_file, 'w') as writer:
        writer.write(json.dumps(all_tag_predictions, indent=4) + '\n')
    return


def _get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
