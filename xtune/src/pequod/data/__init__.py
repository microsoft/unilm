import logging
from transformers.data.processors.utils import InputFeatures


logger = logging.getLogger(__name__)
  

def convert_examples_to_features(
  processor, examples, tokenizer, max_length, label_list,
  pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True):
  
  if label_list is None: label_list = processor.get_labels()

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for ex_index, example in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % ex_index)
    inputs = tokenizer.encode_plus(
      example.text_a,
      example.text_b,
      add_special_tokens=True,
      max_length=max_length)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
  
    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    label = label_map[example.label]
    if ex_index < 3:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      logger.info("label: %s (id = %d)" % (example.label, label))
    
    features.append(InputFeatures(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      label=label))
    
  return features