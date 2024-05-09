import os
import random
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

OURS_TEMPLATE = "There is a special magic number inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the magic number there. {context} "
RANDOM_NEEDLE_CITIES  = [
    'Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
    'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
    'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
    'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
    'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
    'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
    'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
    'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
    'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
    'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
    'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
    'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta'
]
QUESTION_TEMPLATE = "What is the special magic {city} number? The special magic {city} number is "
# NEEDLE_TEMPLATE = "The special magic {city} number is {rnd_number} . Remember it. The special magic {city} number is {rnd_number} . "
NEEDLE_TEMPLATE = "The special magic {city} number is {rnd_number} . "
@dataclass
class NeedleHaystackEvalConfig(FairseqDataclass):
    max_len_b: int = field(
        default=5,
        metadata={"help":"max_len_b"}
    )
    tokens_per_sample: int = field(
        default=16384,
    )
    interval: int = field(
        default=1024,
    )
    needle_file_path: str = field(
        default="/mnt/msranlp/yutao/data/PaulGrahamEssays",
    )

@register_criterion("needle_haystack", dataclass=NeedleHaystackEvalConfig)
class NeedleHaystackEvalCriterion(FairseqCriterion):
    def __init__(self, cfg: NeedleHaystackEvalConfig, task):
        super().__init__(task)
        self.cfg = cfg
        self.essay_list = os.listdir(cfg.needle_file_path) * 5000

    def generate_garbage(self, length):
        current_text = ""
        current_length = 0
        while True:
            essay = random.choice(self.essay_list)
            essay = open(os.path.join(self.cfg.needle_file_path, essay)).read().splitlines()
            for line in essay:
                tokens = self.task.tokenizer.encode(line + " ")
                if current_length + len(tokens) > length:
                    return current_text
                current_text += line + " "
                current_length += len(tokens)

    def generate_prompt_landmark(self, prefix_length, suffix_length):
        """Generates a text file and inserts an passkey at a random position."""
        city = random.choice(RANDOM_NEEDLE_CITIES)
        magic_number = random.randint(1, 50000)
        garbage_prefix = self.generate_garbage(prefix_length)
        garbage_suffix = self.generate_garbage(suffix_length)
        information_line = NEEDLE_TEMPLATE.format(city=city, rnd_number=magic_number)
        final_question = QUESTION_TEMPLATE.format(city=city)
        lines = [
            garbage_prefix,
            information_line,
            garbage_suffix,
            final_question,
        ]
        context = "\n".join(lines)
        return OURS_TEMPLATE.format(context=context), str(magic_number)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model.eval()
        all_retrieval_result = {}
        random.seed(0)
        for context_length in range(self.cfg.interval, self.cfg.tokens_per_sample + 1, self.cfg.interval):
            all_length = (context_length - 150)
            local_retrieval_result = []
            depth_number = 10
            for depth_ratio in range(0, depth_number + 1):
                prefix_length = int(all_length * depth_ratio / depth_number)
                suffix_length = all_length - prefix_length
                n_correct = 0
                times = 10
                for _ in range(times):
                    prompt, pass_key = self.generate_prompt_landmark(prefix_length, suffix_length)
                    prompt_tokens = self.task.tokenizer.encode(prompt, bos=True)
                    prompt_tokens = torch.tensor([prompt_tokens], device="cuda")
                    print(prompt_tokens.shape)
                    output = self.generate(model, prompt_tokens)
                    pred = self.task.tokenizer.decode(output[0, prompt_tokens.shape[1]:])
                    print("Answer: ", pass_key)
                    print("Pred: ", pred)
                    if pass_key in pred:
                        n_correct += 1
                local_retrieval_result.append(n_correct / times)
            all_retrieval_result[context_length] = local_retrieval_result

        print(all_retrieval_result)
        return 0, 1, {"loss": 0}

    def generate(self, model, net_input, generate_tokens=20, chunk_length = 32768):
        output_tokens = torch.cat((net_input, torch.full((net_input.shape[0], generate_tokens), self.task.tokenizer.pad_id).long().cuda()), dim=1)
        begin_pad_index = torch.where(output_tokens == self.task.tokenizer.pad_id)[1].min().item()
        incremental_state = {}
        eos_reached = torch.tensor([False] * net_input.shape[0], device="cuda")
        # prefilling
        for begin_index in range(0, begin_pad_index - 1, chunk_length):
            end_index = min(begin_index + chunk_length, begin_pad_index - 1)
            _, _ = model(output_tokens[:, begin_index : end_index], incremental_state=incremental_state, start_pos=begin_index, skip_cross_decoder=True, is_prefilling=True)
        # generation
        for index in range(begin_pad_index, output_tokens.shape[1]):
            generation_net_output, _ = model(output_tokens[:, index - 1].unsqueeze(-1), incremental_state=incremental_state, start_pos=index - 1, skip_cross_decoder=False, is_prefilling=False)
            generation_net_output[:, :, self.task.tokenizer.bos_id] = -math.inf
            generation_net_output[:, :, self.task.tokenizer.pad_id] = -math.inf
            next_tokens = torch.argmax(generation_net_output[:, -1, :], dim=-1)
            pad_tokens = output_tokens[:, index]
            next_tokens = torch.where((pad_tokens == self.task.tokenizer.pad_id) & ~eos_reached, next_tokens, pad_tokens)
            output_tokens[:, index] = next_tokens
            eos_reached |= (
                next_tokens == self.task.tokenizer.eos_id
            )
            if all(eos_reached):
                break
            
        return output_tokens
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        metric_sum = sum(log.get("metric", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / ntokens, ntokens, round=3
        )
        metrics.log_scalar(
            "metric", metric_sum / nsentences, nsentences, round=3
        )


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True