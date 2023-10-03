import numpy as np
import torch
import random
import gradio as gr

controlnet_example = [
    ['appimg/doctor.jpg', '<i>', 'appimg/bengio.jpg', None],
    ['appimg/doctor.jpg', '<i> as an oil painting in the style of <i>', 'appimg/bengio.jpg', 'appimg/vangogh.jpg'],
]

BOI_SYMBOL = "<image>"
EOI_SYMBOL = "</image>"
MIN_SEED = 0
MAX_SEED = np.iinfo(np.int32).max
MAX_COLORS = 12
MAX_INPUT_IMAGES = 10
DEFAULT_INPUT_IMAGES = 2
MAX_IMAGES_PER_PROMPT = 4
DEFAULT_IMAGES_PER_PROMPT = 1

MIN_IMAGE_RESOLUTION = 256
MAX_IMAGE_RESOLUTION = 768
DEFAULT_IMAGE_RESOLUTION = 768

CONTROLNET_MODEL_IDS = {
    'Openpose': 'lllyasviel/control_v11p_sd15_openpose',
    'Canny': 'lllyasviel/control_v11p_sd15_canny',
    'MLSD': 'lllyasviel/control_v11p_sd15_mlsd',
    'scribble': 'lllyasviel/control_v11p_sd15_scribble',
    'softedge': 'lllyasviel/control_v11p_sd15_softedge',
    'segmentation': 'lllyasviel/control_v11p_sd15_seg',
    'depth': 'lllyasviel/control_v11f1p_sd15_depth',
    'NormalBae': 'lllyasviel/control_v11p_sd15_normalbae',
    'lineart': 'lllyasviel/control_v11p_sd15_lineart',
    'lineart_anime': 'lllyasviel/control_v11p_sd15s2_lineart_anime',
    'shuffle': 'lllyasviel/control_v11e_sd15_shuffle',
    'ip2p': 'lllyasviel/control_v11e_sd15_ip2p',
    'inpaint': 'lllyasviel/control_v11e_sd15_inpaint',
}

canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data, canvasData]
}
"""

set_canvas_size = """
async (aspect) => {
  if(aspect ==='square'){
    _updateCanvas(512,512)
  }
  if(aspect ==='horizontal'){
    _updateCanvas(768,512)
  }
  if(aspect ==='vertical'){
    _updateCanvas(512,768)
  }
}
"""


def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def variable_images(k):
    k = int(k)
    return [gr.Textbox.update(visible=True)] * k + [gr.Textbox.update(visible=False)] * (MAX_INPUT_IMAGES - k)
