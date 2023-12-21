MODEL_NAME_TO_POOL_TYPE = {
    'e5-small': 'avg',
    'e5-base': 'avg',
    'e5-large': 'avg',
    'e5-small-v2': 'avg',
    'e5-base-v2': 'avg',
    'e5-large-v2': 'avg',
    'multilingual-e5-small': 'avg',
    'multilingual-e5-base': 'avg',
    'multilingual-e5-large': 'avg',
    'e5-mistral-7b-instruct': 'last',
}


MODEL_NAME_TO_PREFIX_TYPE = {
    'e5-small': 'query_or_passage',
    'e5-base': 'query_or_passage',
    'e5-large': 'query_or_passage',
    'e5-small-v2': 'query_or_passage',
    'e5-base-v2': 'query_or_passage',
    'e5-large-v2': 'query_or_passage',
    'multilingual-e5-small': 'query_or_passage',
    'multilingual-e5-base': 'query_or_passage',
    'multilingual-e5-large': 'query_or_passage',
    'e5-mistral-7b-instruct': 'instruction',
}
