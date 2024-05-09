import json
import numpy as np

class HarnessBaseTask:
    def __init__(self, tokenizer, data_dir, tokens_per_sample=1024):
        self.tokenizer = tokenizer
        self.class_num = 1
        self.tokens_per_sample = tokens_per_sample
        self.base_dir = data_dir
        self.set_dataname()
        self.set_class_num()
        self.dataset = self.load_data()

    def load_data(self):
        import os
        datasets = []
        with open(os.path.join(self.base_dir, self.dataname), "r", encoding='utf-8') as fin:
            for line in fin:
                obj = json.loads(line)
                datasets.append(
                    {
                        "text": obj["ctx"] if "ctx" in obj else None,
                        "label": obj["label"] if "label" in obj else None,
                        "choices": obj["choices"] if "choices" in obj else [],
                        "gold": obj["gold"] if "gold" in obj else None,
                        "raw": obj,
                    }
                )
        return datasets

    def set_class_num(self):
        raise NotImplementedError
    
    def set_dataname(self):
        raise NotImplementedError

    def preprocess_example(self, example):
        raise NotImplementedError

    def get_data_for_evaluation(self):
        src_tokens = []
        gpt_loss_mask = []
        label_length = []
        labels = []
        cut_num = 0
        for i, example in enumerate(self.dataset):
            input_str, label_str, label = self.preprocess_example(example)
            if i < 2:
                print(f"input str is {input_str}")
                print(f"label str is {label_str}")

            for j in range(len(input_str)):
                sub_input_str, sub_label_str = input_str[j], label_str[j]
                input_token = self.tokenizer.encode(sub_input_str)
                label_token = self.tokenizer.encode(sub_input_str + sub_label_str)[len(input_token):]
                if len(input_token) + len(label_token) + 1 >= self.tokens_per_sample:
                    cut_num += 1
                    input_token = input_token[-(self.tokens_per_sample - len(label_token) - 1):]

                src_tokens.append([self.tokenizer.bos_id] + input_token + label_token)
                gpt_loss_mask.append([False] * (len(input_token) + 1) + [True] * len(label_token))
                label_length.append(len(sub_label_str.strip()))
                labels.append(label)

        if cut_num > 0:
            print(f"cut {cut_num} examples")
            
        return np.array(src_tokens), np.array(gpt_loss_mask), np.array(label_length), np.array(labels)
    

class HarnessAnlir1(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 3

    def set_dataname(self):
        self.dataname = "anli_r1"

    def preprocess_example(self, example):
        input_str = [example["text"]] * self.class_num
        answer_str = [" True", " Neither", " False"]
        label = example["label"]
        return input_str, answer_str, label

class HarnessAnlir2(HarnessAnlir1):
    def set_dataname(self):
        self.dataname = "anli_r2"

class HarnessAnlir3(HarnessAnlir1):
    def set_dataname(self):
        self.dataname = "anli_r3"

class HarnessArc_challenge(HarnessBaseTask):
    '''
    using harness to evaluate arc challenge
    '''
    def set_class_num(self):
        self.class_num = 5

    def set_dataname(self):
        self.dataname = "arc_challenge"

    def preprocess_example(self, example):
        input_str = [example["text"]] * len(example["choices"])    
        answer_str = [' ' + item for item in example["choices"]]
        label = example["gold"]
        return input_str, answer_str, label

class HarnessArc_challenge25s(HarnessBaseTask):
    '''
    using harness to evaluate arc challenge
    '''
    def set_class_num(self):
        self.class_num = 5

    def set_dataname(self):
        self.dataname = "arc_challenge_25s"

    def preprocess_example(self, example):
        input_str = [example["text"]] * len(example["choices"])    
        answer_str = [' ' + item for item in example["choices"]]
        label = example["gold"]
        return input_str, answer_str, label

class HarnessArc_easy(HarnessArc_challenge):
    def set_class_num(self):
        self.class_num = 5

    def set_dataname(self):
        self.dataname = "arc_easy"

class HarnessBoolq(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 2

    def set_dataname(self):
        self.dataname = "boolq"

    def preprocess_example(self, example):
        input_str = [example["text"]] * self.class_num
        answer_str = [" no", " yes"]
        label = example["label"]
        return input_str, answer_str, label

class HarnessCopa(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 2

    def set_dataname(self):
        self.dataname = "copa"

    def preprocess_example(self, example):
        input_str = [example["text"]] * self.class_num
        answer_str = [' ' + example['raw']['choice1'], ' ' + example['raw']['choice2']]
        label = example["label"]
        return input_str, answer_str, label

class HarnessOpenbookqa(HarnessArc_challenge):
    def set_class_num(self):
        self.class_num = 4

    def set_dataname(self):
        self.dataname = "openbookqa"

class HarnessPiqa(HarnessArc_challenge):
    def set_class_num(self):
        self.class_num = 2

    def set_dataname(self):
        self.dataname = "piqa"

class HarnessRte(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 2

    def set_dataname(self):
        self.dataname = "rte"

    def preprocess_example(self, example):
        input_str = [example["text"]] * self.class_num
        answer_str = [' True', ' False']
        label = example["label"]
        return input_str, answer_str, label

class HarnessWic(HarnessRte):
    def set_dataname(self):
        self.dataname = "wic"

class HarnessWinogrande(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 2

    def set_dataname(self):
        self.dataname = "winogrande"

    def preprocess_example(self, example):
        pronoun_loc = example['raw']['sentence'].index("_")
        input_str = []
        input_str.append(example['raw']['sentence'][:pronoun_loc].strip() + ' ' + example['raw']['option1'])
        input_str.append(example['raw']['sentence'][:pronoun_loc].strip() + ' ' + example['raw']['option2'])
        answer_str = [" " + example['raw']["sentence"][pronoun_loc + 1:].strip()] * self.class_num
        label = int(example['raw']['answer']) - 1
        return input_str, answer_str, label

class HarnessHellaswag(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 4

    def set_dataname(self):
        self.dataname = "hellaswag"

    def preprocess_example(self, example):
        input_str = [example["text"]] * self.class_num
        answer_str = [' ' + item for item in example["choices"]]
        label = example["gold"]
        return input_str, answer_str, label


class HarnessHellaswag10s(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 4

    def set_dataname(self):
        self.dataname = "hellaswag_10s"

    def preprocess_example(self, example):
        input_str = [example["text"]] * self.class_num
        answer_str = [' ' + item for item in example["choices"]]
        label = example["gold"]
        return input_str, answer_str, label


class HarnessTruthfullqaMC1(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 1

    def set_dataname(self):
        self.dataname = "truthfulqa_mc"

    def preprocess_example(self, example):
        input_str = [example["text"]] * len(example["raw"]["mc1_targets"]["choices"])
        answer_str = [' ' + item for item in example["raw"]["mc1_targets"]["choices"]]
        label = 0 # dummy label
        return input_str, answer_str, label
    


class HarnessTruthfullqaMC2(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 1

    def set_dataname(self):
        self.dataname = "truthfulqa_mc"

    def preprocess_example(self, example):
        input_str = [example["text"]] * len(example["raw"]["mc2_targets"]["choices"])
        answer_str = [' ' + item for item in example["raw"]["mc2_targets"]["choices"]]
        label = 0 # dummy label
        return input_str, answer_str, label
    

class HarnessRecord(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 1

    def set_dataname(self):
        self.dataname = "record"

    def preprocess_example(self, example):
        input_str = [example["text"]] * len(example["raw"]["entities"])
        answer_str = [f'  - {example["raw"]["query"]}'.replace("@placeholder", item) for item in example["raw"]["entities"]]
        label = 0 # dummy label
        return input_str, answer_str, label

class HarnessSCIQ(HarnessBaseTask):
    def set_class_num(self):
        self.class_num = 4

    def set_dataname(self):
        self.dataname = "sciq"

    def preprocess_example(self, example):
        input_str = [example["text"]] * self.class_num
        answer_str = [' ' + example["raw"]["distractor1"], 
                      ' ' + example["raw"]["distractor2"], 
                      ' ' + example["raw"]["distractor3"], 
                      ' ' + example["raw"]["correct_answer"]
                    ]
        label = 3
        return input_str, answer_str, label