import tqdm
from torch.utils.data import Dataset
from markuplmft.data.tag_utils import tags_dict
import pickle
import os
import constants


class SwdeFeature(object):
    def __init__(self,
                 html_path,
                 input_ids,
                 token_type_ids,
                 attention_mask,
                 xpath_tags_seq,
                 xpath_subs_seq,
                 labels,
                 involved_first_tokens_pos,
                 involved_first_tokens_xpaths,
                 involved_first_tokens_types,
                 involved_first_tokens_text,
                 ):
        """
        html_path: indicate which page the feature belongs to
        input_ids: RT
        token_type_ids: RT
        attention_mask: RT
        xpath_tags_seq: RT
        xpath_subs_seq: RT
        labels: RT
        involved_first_tokens_pos: a list, indicate the positions of the first-tokens in this feature
        involved_first_tokens_xpaths: the xpaths of the first-tokens, used to build dict
        involved_first_tokens_types: the types of the first-tokens
        involved_first_tokens_text: the text of the first tokens

        Note that `involved_xxx` are not fixed-length array, so they shouldn't be sent into our model
        They are just used for evaluation
        """
        self.html_path = html_path
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.xpath_tags_seq = xpath_tags_seq
        self.xpath_subs_seq = xpath_subs_seq
        self.labels = labels
        self.involved_first_tokens_pos = involved_first_tokens_pos
        self.involved_first_tokens_xpaths = involved_first_tokens_xpaths
        self.involved_first_tokens_types = involved_first_tokens_types
        self.involved_first_tokens_text = involved_first_tokens_text


class SwdeDataset(Dataset):
    def __init__(self,
                 all_input_ids,
                 all_attention_mask,
                 all_token_type_ids,
                 all_xpath_tags_seq,
                 all_xpath_subs_seq,
                 all_labels=None,
                 ):
        '''
        print(type(all_input_ids))
        print(type(all_attention_mask))
        print(type(all_token_type_ids))
        print(type(all_xpath_tags_seq))
        print(type(all_xpath_subs_seq))
        print(type(all_labels))
        raise ValueError
        '''
        self.tensors = [all_input_ids, all_attention_mask, all_token_type_ids,
                        all_xpath_tags_seq, all_xpath_subs_seq]

        if not all_labels is None:
            self.tensors.append(all_labels)

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)


def process_xpath(xpath: str):
    if xpath.endswith("/tail"):
        xpath = xpath[:-5]

    xpath_tags_seq, xpath_subs_seq = [], []
    units = xpath.split("/")

    for unit in units:
        if not unit:
            continue
        if '[' not in unit:
            xpath_tags_seq.append(tags_dict.get(unit, 215))
            xpath_subs_seq.append(0)
        else:
            xx = unit.split('[')
            name = xx[0]
            id = int(xx[1][:-1])
            xpath_tags_seq.append(tags_dict.get(name, 215))
            xpath_subs_seq.append(min(id, 1000))

    assert len(xpath_subs_seq) == len(xpath_tags_seq)

    if len(xpath_tags_seq) > 50:
        xpath_tags_seq = xpath_tags_seq[-50:]
        xpath_subs_seq = xpath_subs_seq[-50:]

    xpath_tags_seq = xpath_tags_seq + [216] * (50 - len(xpath_tags_seq))
    xpath_subs_seq = xpath_subs_seq + [1001] * (50 - len(xpath_subs_seq))

    return xpath_tags_seq, xpath_subs_seq


def get_swde_features(root_dir, vertical, website, tokenizer,
                      doc_stride, max_length, prev_nodes, n_pages):
    real_max_token_num = max_length - 2  # for cls and sep
    padded_xpath_tags_seq = [216] * 50
    padded_xpath_subs_seq = [1001] * 50

    filename = os.path.join(root_dir, f"{vertical}-{website}-{n_pages}.pickle")
    with open(filename, "rb") as f:
        raw_data = pickle.load(f)

    features = []

    for index in tqdm.tqdm(raw_data, desc=f"Processing {vertical}-{website}-{n_pages} features ..."):
        html_path = f"{vertical}-{website}-{index}.htm"
        needed_docstrings_id_set = set()
        for i in range(len(raw_data[index])):
            doc_string_type = raw_data[index][i][2]
            if doc_string_type == "fixed-node":
                continue
            # we take i-3, i-2, i-1 into account

            needed_docstrings_id_set.add(i)

            used_prev = 0
            prev_id = i - 1
            while prev_id >= 0 and used_prev < prev_nodes:
                if raw_data[index][prev_id][0].strip():
                    needed_docstrings_id_set.add(prev_id)
                    used_prev += 1
                prev_id -= 1

        needed_docstrings_id_list = sorted(list(needed_docstrings_id_set))

        all_token_ids_seq = []
        all_xpath_tags_seq = []
        all_xpath_subs_seq = []
        token_to_ori_map_seq = []
        all_labels_seq = []

        first_token_pos = []
        first_token_xpaths = []
        first_token_type = []
        first_token_text = []

        for i, needed_id in enumerate(needed_docstrings_id_list):
            text = raw_data[index][needed_id][0]
            xpath = raw_data[index][needed_id][1]
            type = raw_data[index][needed_id][2]
            token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            xpath_tags_seq, xpath_subs_seq = process_xpath(xpath)

            all_token_ids_seq += token_ids
            all_xpath_tags_seq += [xpath_tags_seq] * len(token_ids)
            all_xpath_subs_seq += [xpath_subs_seq] * len(token_ids)

            token_to_ori_map_seq += [i] * len(token_ids)

            if type == "fixed-node":
                all_labels_seq += [-100] * len(token_ids)
            else:
                # we always use the first token to predict
                first_token_pos.append(len(all_labels_seq))
                first_token_type.append(type)
                first_token_xpaths.append(xpath)
                first_token_text.append(text)

                all_labels_seq += [constants.ATTRIBUTES_PLUS_NONE[vertical].index(type)] * len(token_ids)

        assert len(all_token_ids_seq) == len(all_xpath_tags_seq)
        assert len(all_token_ids_seq) == len(all_xpath_subs_seq)
        assert len(all_token_ids_seq) == len(all_labels_seq)

        # we have all the pos of variable nodes in all_token_ids_seq
        # now we need to assign them into each feature

        start_pos = 0
        flag = False

        curr_first_token_index = 0

        while True:

            # invloved is [ start_pos , end_pos )

            token_type_ids = [0] * max_length  # that is always this

            end_pos = start_pos + real_max_token_num
            # add start_pos ~ end_pos as a feature
            splited_token_ids_seq = [tokenizer.cls_token_id] + all_token_ids_seq[start_pos:end_pos] + [
                tokenizer.sep_token_id]
            splited_xpath_tags_seq = [padded_xpath_tags_seq] + all_xpath_tags_seq[start_pos:end_pos] + [
                padded_xpath_tags_seq]
            splited_xpath_subs_seq = [padded_xpath_subs_seq] + all_xpath_subs_seq[start_pos:end_pos] + [
                padded_xpath_subs_seq]
            splited_labels_seq = [-100] + all_labels_seq[start_pos:end_pos] + [-100]

            # locate first-tokens in them
            involved_first_tokens_pos = []
            involved_first_tokens_xpaths = []
            involved_first_tokens_types = []
            involved_first_tokens_text = []

            while curr_first_token_index < len(first_token_pos) \
                    and end_pos > first_token_pos[curr_first_token_index] >= start_pos:
                involved_first_tokens_pos.append(
                    first_token_pos[curr_first_token_index] - start_pos + 1)  # +1 for [cls]
                involved_first_tokens_xpaths.append(first_token_xpaths[curr_first_token_index])
                involved_first_tokens_types.append(first_token_type[curr_first_token_index])
                involved_first_tokens_text.append(first_token_text[curr_first_token_index])
                curr_first_token_index += 1

            # we abort this feature if no useful node in it
            if len(involved_first_tokens_pos) == 0:
                break

            if end_pos >= len(all_token_ids_seq):
                flag = True
                # which means we need to pad in this feature
                current_len = len(splited_token_ids_seq)
                splited_token_ids_seq += [tokenizer.pad_token_id] * (max_length - current_len)
                splited_xpath_tags_seq += [padded_xpath_tags_seq] * (max_length - current_len)
                splited_xpath_subs_seq += [padded_xpath_subs_seq] * (max_length - current_len)
                splited_labels_seq += [-100] * (max_length - current_len)
                attention_mask = [1] * current_len + [0] * (max_length - current_len)

            else:
                # no need to pad, the splited seq is exactly with the length `max_length`
                assert len(splited_token_ids_seq) == max_length
                attention_mask = [1] * max_length

            features.append(
                SwdeFeature(
                    html_path=html_path,
                    input_ids=splited_token_ids_seq,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    xpath_tags_seq=splited_xpath_tags_seq,
                    xpath_subs_seq=splited_xpath_subs_seq,
                    labels=splited_labels_seq,
                    involved_first_tokens_pos=involved_first_tokens_pos,
                    involved_first_tokens_xpaths=involved_first_tokens_xpaths,
                    involved_first_tokens_types=involved_first_tokens_types,
                    involved_first_tokens_text=involved_first_tokens_text,
                )
            )

            start_pos = end_pos - doc_stride

            if flag:
                break

    return features
