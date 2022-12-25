from sacred import Experiment

ex = Experiment("VLMo")


def _loss_names(d):
    ret = {
        "itm": 0, # image-text matching loss
        "itc": 0, # image-text contrastive loss
        "mlm": 0, # masked language modeling loss
        "textmlm": 0, # text-only masked language modeling
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0, # retrieval task ft
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vlmo"
    seed = 1
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "itc": 1, "mlm": 1})
    batch_size = 1024  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    image_size = 224
    draw_false_image = 0
    image_only = False
    text_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    max_text_len_of_initckpt = 196
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    model_arch = "vlmo_base_patch16"
    drop_path_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 200000
    warmup_steps = 0.1
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False
    get_recall_rerank_metric = False
    k_test = 32

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    use_sharded_training = False
    resume_during_training = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 4  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16


# ----------------------- language pretraining config -----------------------


@ex.named_config
def task_textmlm_base():
    exp_name = "textmlm_base"
    datasets = ["wikibk"]
    loss_names = _loss_names({"textmlm": 1})
    batch_size = 1024
    max_text_len = 196
    learning_rate = 2e-4
    whole_word_masking = True
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_textmlm_base_plus():
    exp_name = "textmlm_base_plus"
    datasets = ["wikibk"]
    loss_names = _loss_names({"textmlm": 1})
    batch_size = 1024
    max_text_len = 196
    learning_rate = 2e-4
    whole_word_masking = True
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vlmo_base_plus_patch16"


# ----------------------- vision-language pretraining config -----------------------


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm_itc_base():
    exp_name = "mlm_itm_itc_base"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    batch_size = 1024
    whole_word_masking = True
    learning_rate = 2e-4
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_mlm_itm_itc_base_plus():
    exp_name = "mlm_itm_itc_base_plus"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    batch_size = 1024
    whole_word_masking = True
    learning_rate = 1e-4
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_mlm_itm_itc_large():
    exp_name = "mlm_itm_itc_large"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
    batch_size = 1024
    whole_word_masking = True
    learning_rate = 5e-5
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    model_arch = "vit_large_patch16_224"


# ----------------------- NLVR2 fine-tuning configs -----------------------


@ex.named_config
def task_finetune_nlvr2_base():
    exp_name = "finetune_nlvr2_base"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 5e-5
    val_transform_keys = ["square_transform"]
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_nlvr2_base_plus():
    exp_name = "finetune_nlvr2_base_plus"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.2
    val_transform_keys = ["square_transform"]
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_nlvr2_base_image384():
    exp_name = "finetune_nlvr2_base_image384"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 5e-5
    val_transform_keys = ["square_transform"]
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_nlvr2_base_plus_image384():
    exp_name = "finetune_nlvr2_base_plus_image384"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.2
    val_transform_keys = ["square_transform"]
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_nlvr2_large():
    exp_name = "finetune_nlvr2_large"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


@ex.named_config
def task_finetune_nlvr2_large_image384():
    exp_name = "finetune_nlvr2_large_image384"
    datasets = ["nlvr2"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


# ----------------------- VQAv2 Fine-tuning configs -----------------------


@ex.named_config
def task_finetune_vqa_base_image480():
    exp_name = "finetune_vqa_base_image480"
    datasets = ["vqa"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    lr_mult = 20
    image_size = 480
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_vqa_base_plus_image480():
    exp_name = "finetune_vqa_base_plus_image480"
    datasets = ["vqa"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 3e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    lr_mult = 20
    image_size = 480
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_vqa_large_image480():
    exp_name = "finetune_vqa_large_image480"
    datasets = ["vqa"]
    train_transform_keys = ["square_transform_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    learning_rate = 1.5e-5
    drop_path_rate = 0.15
    val_transform_keys = ["square_transform"]
    lr_mult = 20
    image_size = 480
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


# ----------------------- F30K IR/TR Fine-tuning configs -----------------------


@ex.named_config
def task_finetune_irtr_f30k_base():
    exp_name = "finetune_irtr_f30k_base"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 1500
    warmup_steps = 150
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.15
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_irtr_f30k_base_image384():
    exp_name = "finetune_irtr_f30k_base_image384"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 1500
    warmup_steps = 150
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.15
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_irtr_f30k_base_plus_image384():
    exp_name = "finetune_irtr_f30k_base_plus_image384"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 1500
    warmup_steps = 150
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_irtr_f30k_large_image384():
    exp_name = "finetune_irtr_f30k_large_image384"
    datasets = ["f30k"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 1500
    warmup_steps = 150
    get_recall_metric = True
    learning_rate = 2e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


# ----------------------- COCO IR/TR Fine-tuning configs -----------------------


@ex.named_config
def task_finetune_irtr_coco_base_image384():
    exp_name = "finetune_irtr_coco_base_image384"
    datasets = ["coco"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 3000
    warmup_steps = 300
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_patch16"


@ex.named_config
def task_finetune_irtr_coco_base_plus_image384():
    exp_name = "finetune_irtr_coco_base_plus_image384"
    datasets = ["coco"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 3000
    warmup_steps = 300
    get_recall_metric = True
    learning_rate = 3e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_base_plus_patch16"


@ex.named_config
def task_finetune_irtr_coco_large_image384():
    exp_name = "finetune_irtr_coco_large_image384"
    datasets = ["coco"]
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    loss_names = _loss_names({"irtr": 1.0})
    batch_size = 3072
    max_epoch = 50
    max_steps = 3000
    warmup_steps = 300
    get_recall_metric = True
    learning_rate = 2e-5
    drop_path_rate = 0.2
    image_size = 384
    use_sharded_training=False
    model_arch = "vlmo_large_patch16"


# ----------------------- Other configs -----------------------


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end
@ex.named_config
def step1_5k():
    max_epoch = 100
    warmup_steps = 150
    max_steps = 1500


@ex.named_config
def step3k():
    max_epoch = 100
    warmup_steps = 300
    max_steps = 3000


@ex.named_config
def step200k():
    max_epoch = 200
    warmup_steps = 2500
    max_steps = 200000


@ex.named_config
def step500k():
    max_epoch = 500
    warmup_steps = 2500
    max_steps = 500000