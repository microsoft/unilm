import os
import re
import zipfile
import torch
import gradio as gr

print('hello', gr.__version__)


import time
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, DiffusionPipeline, LCMScheduler
from tqdm import tqdm
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import random
import copy

import string
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '  # len(aphabet) = 95
'''alphabet
0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ 
'''

if not os.path.exists('images2'):
    os.system('wget https://huggingface.co/datasets/JingyeChen22/TextDiffuser/resolve/main/images2.zip')
    with zipfile.ZipFile('images2.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

# os.system('nvidia-smi')
os.system('ls')

#### import m1
from fastchat.model import load_model, get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM
m1_model_path = 'JingyeChen22/textdiffuser2_layout_planner'
# m1_model, m1_tokenizer = load_model(
#     m1_model_path,
#     'cuda',
#     1,
#     None,
#     False,
#     False,
#     revision="main",
#     debug=False,
# )

m1_tokenizer = AutoTokenizer.from_pretrained(m1_model_path, use_fast=False)
m1_model = AutoModelForCausalLM.from_pretrained(
    m1_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
).cuda()

#### import diffusion models
text_encoder = CLIPTextModel.from_pretrained(
    'JingyeChen22/textdiffuser2-full-ft', subfolder="text_encoder"
).cuda().half()
tokenizer = CLIPTokenizer.from_pretrained(
    'runwayml/stable-diffusion-v1-5', subfolder="tokenizer"
)

#### additional tokens are introduced, including coordinate tokens and character tokens
print('***************')
print(len(tokenizer))
for i in range(520):
    tokenizer.add_tokens(['l' + str(i) ]) # left
    tokenizer.add_tokens(['t' + str(i) ]) # top
    tokenizer.add_tokens(['r' + str(i) ]) # width
    tokenizer.add_tokens(['b' + str(i) ]) # height    
for c in alphabet:
    tokenizer.add_tokens([f'[{c}]']) 
print(len(tokenizer))
print('***************')

vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae").half().cuda()
unet = UNet2DConditionModel.from_pretrained(
    'JingyeChen22/textdiffuser2-full-ft', subfolder="unet"
).half().cuda()
text_encoder.resize_token_embeddings(len(tokenizer))


#### load lcm components
model_id = "lambdalabs/sd-pokemon-diffusers"
lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"
pipe = DiffusionPipeline.from_pretrained(model_id, unet=copy.deepcopy(unet), tokenizer=tokenizer, text_encoder=copy.deepcopy(text_encoder), torch_dtype=torch.float16)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lcm_lora_id)
pipe.to(device="cuda")

global_dict = {}
#### for interactive
# stack = []
# state = 0   
font = ImageFont.truetype("./Arial.ttf", 32)

def skip_fun(i, t, guest_id):
    global_dict[guest_id]['state'] = 0
    # global state
    # state = 0


def exe_undo(i, t, guest_id):

    global_dict[guest_id]['stack'] = []
    global_dict[guest_id]['state'] = 0

    # global stack
    # global state
    # state = 0
    # stack = []
    image = Image.open(f'./gray256.jpg')
    # print('stack', stack)
    return image


def exe_redo(i, t, guest_id):
    # global state 
    # state = 0
    global_dict[guest_id]['state'] = 0

    if len(global_dict[guest_id]['stack']) > 0:
        global_dict[guest_id]['stack'].pop()
    image = Image.open(f'./gray256.jpg')
    draw = ImageDraw.Draw(image)

    for items in global_dict[guest_id]['stack']:
        # print('now', items)
        text_position, t = items
        if len(text_position) == 2:
            x, y = text_position
            text_color = (255, 0, 0)  
            draw.text((x+2, y), t, font=font, fill=text_color)
            r = 4
            leftUpPoint = (x-r, y-r)
            rightDownPoint = (x+r, y+r)
            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
        elif len(text_position) == 4:
            x0, y0, x1, y1 = text_position
            text_color = (255, 0, 0)  
            draw.text((x0+2, y0), t, font=font, fill=text_color)
            r = 4
            leftUpPoint = (x0-r, y0-r)
            rightDownPoint = (x0+r, y0+r)
            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
            draw.rectangle((x0,y0,x1,y1), outline=(255, 0, 0) )

    print('stack', global_dict[guest_id]['stack'])
    return image

def get_pixels(i, t, guest_id, evt: gr.SelectData):
    # global state

    # register
    if guest_id == '-1':
        seed = str(int(time.time()))
        global_dict[str(seed)] = {
            'state': 0,
            'stack': []
        }
        guest_id = str(seed)
    else:
        seed = guest_id

    text_position = evt.index

    if global_dict[guest_id]['state'] == 0:
        global_dict[guest_id]['stack'].append(
            (text_position, t)
        )
        print(text_position, global_dict[guest_id]['stack'])
        global_dict[guest_id]['state'] = 1
    else:
        
        (_, t) = global_dict[guest_id]['stack'].pop()
        x, y = _
        global_dict[guest_id]['stack'].append(
            ((x,y,text_position[0],text_position[1]), t)
        )
        global_dict[guest_id]['state'] = 0


    image = Image.open(f'./gray256.jpg')
    draw = ImageDraw.Draw(image)

    for items in global_dict[guest_id]['stack']:
        # print('now', items)
        text_position, t = items
        if len(text_position) == 2:
            x, y = text_position
            text_color = (255, 0, 0)  
            draw.text((x+2, y), t, font=font, fill=text_color)
            r = 4
            leftUpPoint = (x-r, y-r)
            rightDownPoint = (x+r, y+r)
            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
        elif len(text_position) == 4:
            x0, y0, x1, y1 = text_position
            text_color = (255, 0, 0)  
            draw.text((x0+2, y0), t, font=font, fill=text_color)
            r = 4
            leftUpPoint = (x0-r, y0-r)
            rightDownPoint = (x0+r, y0+r)
            draw.ellipse((leftUpPoint,rightDownPoint), fill='red')
            draw.rectangle((x0,y0,x1,y1), outline=(255, 0, 0) )

    print('stack', global_dict[guest_id]['stack'])

    return image, seed


font_layout = ImageFont.truetype('./Arial.ttf', 16)

def get_layout_image(ocrs):

    blank = Image.new('RGB', (256,256), (0,0,0))
    draw = ImageDraw.ImageDraw(blank)

    for line in ocrs.split('\n'):
        line = line.strip()

        if len(line) == 0:
            break

        pred = ' '.join(line.split()[:-1])
        box = line.split()[-1]
        l, t, r, b = [int(i)*2 for i in box.split(',')] # the size of canvas is 256x256
        draw.rectangle([(l, t), (r, b)], outline ="red")
        draw.text((l, t), pred, font=font_layout)
    
    return blank



def text_to_image(guest_id, prompt,keywords,positive_prompt,radio,slider_step,slider_guidance,slider_batch,slider_temperature,slider_natural):

    print(f'[info] Prompt: {prompt} | Keywords: {keywords} | Radio: {radio} | Steps: {slider_step} | Guidance: {slider_guidance} | Natural: {slider_natural}')

    # global stack
    # global state

    if len(positive_prompt.strip()) != 0:
        prompt += positive_prompt

    with torch.no_grad():
        time1 = time.time()
        user_prompt = prompt

        if slider_natural:
            user_prompt = f'{user_prompt}'
            composed_prompt = user_prompt
            prompt = tokenizer.encode(user_prompt)
            layout_image = None
        else:
            if guest_id not in global_dict or len(global_dict[guest_id]['stack']) == 0:

                if len(keywords.strip()) == 0:
                    template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. All keywords are included in the caption. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {user_prompt}'
                else:
                    keywords = keywords.split('/')
                    keywords = [i.strip() for i in keywords]
                    template = f'Given a prompt that will be used to generate an image, plan the layout of visual text for the image. The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, including the coordinates of top, left, right, and bottom. In addition, we also provide all keywords at random order for reference. You dont need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. So let us begin. Prompt: {prompt}. Keywords: {str(keywords)}'

                msg = template
                conv = get_conversation_template(m1_model_path)
                conv.append_message(conv.roles[0], msg)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                inputs = m1_tokenizer([prompt], return_token_type_ids=False)
                inputs = {k: torch.tensor(v).to('cuda') for k, v in inputs.items()}
                output_ids = m1_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=slider_temperature,
                    repetition_penalty=1.0,
                    max_new_tokens=512,
                )

                if m1_model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
                outputs = m1_tokenizer.decode(
                    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                )
                print(f"[{conv.roles[0]}]\n{msg}")
                print(f"[{conv.roles[1]}]\n{outputs}")
                layout_image = get_layout_image(outputs)

                ocrs = outputs.split('\n')
                time2 = time.time()
                print(time2-time1)
                
                # user_prompt = prompt
                current_ocr = ocrs


                ocr_ids = [] 
                print('user_prompt', user_prompt)
                print('current_ocr', current_ocr)
                

                for ocr in current_ocr:
                    ocr = ocr.strip()

                    if len(ocr) == 0 or '###' in ocr or '.com' in ocr:
                        continue

                    items = ocr.split()
                    pred = ' '.join(items[:-1])
                    box = items[-1]
                
                    l,t,r,b = box.split(',')
                    l,t,r,b = int(l), int(t), int(r), int(b)
                    ocr_ids.extend(['l'+str(l), 't'+str(t), 'r'+str(r), 'b'+str(b)])

                    char_list = list(pred)
                    char_list = [f'[{i}]' for i in char_list]
                    ocr_ids.extend(char_list)
                    ocr_ids.append(tokenizer.eos_token_id)     

                caption_ids = tokenizer(
                    user_prompt, truncation=True, return_tensors="pt"
                ).input_ids[0].tolist() 

                try:
                    ocr_ids = tokenizer.encode(ocr_ids)
                    prompt = caption_ids + ocr_ids
                except:
                    prompt = caption_ids

                user_prompt = tokenizer.decode(prompt)
                composed_prompt = tokenizer.decode(prompt)
            
            else:
                user_prompt += ' <|endoftext|><|startoftext|>'
                layout_image = None
                
                for items in global_dict[guest_id]['stack']:
                    position, text = items

                    
                    if len(position) == 2:
                        x, y = position
                        x = x // 4
                        y = y // 4
                        text_str = ' '.join([f'[{c}]' for c in list(text)])
                        user_prompt += f' l{x} t{y} {text_str} <|endoftext|>'
                    elif len(position) == 4:
                        x0, y0, x1, y1 = position
                        x0 = x0 // 4
                        y0 = y0 // 4
                        x1 = x1 // 4
                        y1 = y1 // 4
                        text_str = ' '.join([f'[{c}]' for c in list(text)])
                        user_prompt += f' l{x0} t{y0} r{x1} b{y1} {text_str} <|endoftext|>'

                    # composed_prompt = user_prompt
                    prompt = tokenizer.encode(user_prompt)
                    composed_prompt = tokenizer.decode(prompt)

        prompt = prompt[:77]
        while len(prompt) < 77: 
            prompt.append(tokenizer.pad_token_id) 

        if radio == 'TextDiffuser-2':
            
            prompts_cond = prompt
            prompts_nocond = [tokenizer.pad_token_id]*77

            prompts_cond = [prompts_cond] * slider_batch
            prompts_nocond = [prompts_nocond] * slider_batch

            prompts_cond = torch.Tensor(prompts_cond).long().cuda()
            prompts_nocond = torch.Tensor(prompts_nocond).long().cuda()

            scheduler = DDPMScheduler.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="scheduler") 
            scheduler.set_timesteps(slider_step) 
            noise = torch.randn((slider_batch, 4, 64, 64)).to("cuda").half()
            input = noise

            encoder_hidden_states_cond = text_encoder(prompts_cond)[0].half()
            encoder_hidden_states_nocond = text_encoder(prompts_nocond)[0].half()


            for t in tqdm(scheduler.timesteps):
                with torch.no_grad():  # classifier free guidance
                    noise_pred_cond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_cond[:slider_batch]).sample # b, 4, 64, 64
                    noise_pred_uncond = unet(sample=input, timestep=t, encoder_hidden_states=encoder_hidden_states_nocond[:slider_batch]).sample # b, 4, 64, 64
                    noisy_residual = noise_pred_uncond + slider_guidance * (noise_pred_cond - noise_pred_uncond) # b, 4, 64, 64     
                    input = scheduler.step(noisy_residual, t, input).prev_sample
                    del noise_pred_cond
                    del noise_pred_uncond

                    torch.cuda.empty_cache()

            # decode
            input = 1 / vae.config.scaling_factor * input 
            images = vae.decode(input, return_dict=False)[0] 
            width, height = 512, 512
            results = []
            new_image = Image.new('RGB', (2*width, 2*height))
            for index, image in enumerate(images.cpu().float()):
                image = (image / 2 + 0.5).clamp(0, 1).unsqueeze(0)
                image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                image = Image.fromarray((image * 255).round().astype("uint8")).convert('RGB')
                results.append(image)
                row = index // 2
                col = index % 2
                new_image.paste(image, (col*width, row*height))
            # os.system('nvidia-smi')
            torch.cuda.empty_cache()
            # os.system('nvidia-smi')
            return tuple(results),  composed_prompt, layout_image
        
        elif radio == 'TextDiffuser-2-LCM':
            generator = torch.Generator(device=pipe.device).manual_seed(random.randint(0,1000))
            image = pipe(
                prompt=user_prompt,
                generator=generator,
                # negative_prompt=negative_prompt,
                num_inference_steps=slider_step,
                guidance_scale=1,
                # num_images_per_prompt=slider_batch,
            ).images
            # os.system('nvidia-smi')
            torch.cuda.empty_cache()
            # os.system('nvidia-smi')
            return tuple(image), composed_prompt, layout_image
        
with gr.Blocks() as demo:


    # guest_id = random.randint(0,100000000)
    # register


    gr.HTML(
        """
        <div style="text-align: center; max-width: 1600px; margin: 20px auto;">
        <h2 style="font-weight: 900; font-size: 2.3rem; margin: 0rem">
            TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering
        </h2>
        <h2 style="font-weight: 460; font-size: 1.1rem; margin: 0rem">
            <a href="https://jingyechen.github.io/">Jingye Chen</a>, <a href="https://hypjudy.github.io/website/">Yupan Huang</a>, <a href="https://scholar.google.com/citations?user=0LTZGhUAAAAJ&hl=en">Tengchao Lv</a>, <a href="https://www.microsoft.com/en-us/research/people/lecu/">Lei Cui</a>, <a href="https://cqf.io/">Qifeng Chen</a>, <a href="https://thegenerality.com/">Furu Wei</a>
        </h2>      
        <h2 style="font-weight: 460; font-size: 1.1rem; margin: 0rem">
            HKUST, Sun Yat-sen University, Microsoft Research
        </h2>  
        <h3 style="font-weight: 450; font-size: 1rem; margin: 0rem"> 
        [<a href="https://arxiv.org/abs/2311.16465" style="color:blue;">arXiv</a>] 
        [<a href="https://github.com/microsoft/unilm/tree/master/textdiffuser-2" style="color:blue;">Code</a>]
        [<a href="https://jingyechen.github.io/textdiffuser2/" style="color:blue;">Project Page</a>]
        [<a href="https://discord.gg/q7eHPupu" style="color:purple;">Discord</a>]
        </h3> 
        <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        We propose <b>TextDiffuser-2</b>, aiming at unleashing the power of language models for text rendering. Specifically, we <b>tame a language model into a layout planner</b> to transform user prompt into a layout using the caption-OCR pairs. The language model demonstrates flexibility and automation by inferring keywords from user prompts or incorporating user-specified keywords to determine their positions. Secondly, we <b>leverage the language model in the diffusion model as the layout encoder</b> to represent the position and content of text at the line level. This approach enables diffusion models to generate text images with broader diversity.
        </h2>
        <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
        👀 <b>Tips for using this demo</b>: <b>(1)</b> Please carefully read the disclaimer in the below. Current verison can only support English. <b>(2)</b> The specification of keywords is optional. If provided, the language model will do its best to plan layouts using the given keywords. <b>(3)</b> If a template is given, the layout planner (M1) is not used. <b>(4)</b> Three operations, including redo, undo, and skip are provided. When using skip, only the left-top point of a keyword will be recorded, resulting in more diversity but sometimes decreasing the accuracy. <b>(5)</b> The layout planner can produce different layouts. You can increase the temperature to enhance the diversity. ✨ <b>(6)</b> We also provide the experimental demo combining <b>TextDiffuser-2</b> and <b>LCM</b>. The inference is fast using less sampling steps, although the precision in text rendering might decrease.
        </h2>
        <img src="https://raw.githubusercontent.com/JingyeChen/jingyechen.github.io/master/textdiffuser2/static/images/architecture_blank.jpg" alt="textdiffuser-2">
        </div>
        """)

    with gr.Tab("Text-to-Image"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt. You can let language model automatically identify keywords, or provide them below", placeholder="A beautiful city skyline stamp of Shanghai")
                keywords = gr.Textbox(label="(Optional) Keywords. Should be seperated by / (e.g., keyword1/keyword2/...)", placeholder="keyword1/keyword2")
                positive_prompt = gr.Textbox(label="(Optional) Positive prompt", value=", digital art, very detailed, fantasy, high definition, cinematic light, dnd, trending on artstation")

                # many encounter concurrent problem
                with gr.Accordion("(Optional) Template - Click to paint", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            i = gr.Image(label="Canvas", type='filepath', value=f'./gray256.jpg', height=256, width=256)
                        with gr.Column(scale=1):
                            t = gr.Textbox(label="Keyword", value='input_keyword')
                            redo = gr.Button(value='Redo - Cancel the last keyword') 
                            undo = gr.Button(value='Undo - Clear the canvas') 
                            skip_button = gr.Button(value='Skip - Operate the next keyword') 


                radio = gr.Radio(["TextDiffuser-2", "TextDiffuser-2-LCM"], label="Choice of models", value="TextDiffuser-2")
                slider_natural = gr.Checkbox(label="Natural image generation", value=False, info="The text position and content info will not be incorporated.")
                slider_step = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Sampling step", info="The sampling step for TextDiffuser-2. You may decease the step to 4 when using LCM.")
                slider_guidance = gr.Slider(minimum=1, maximum=13, value=7.5, step=0.5, label="Scale of classifier-free guidance", info="The scale of cfg and is set to 7.5 in default. When using LCM, cfg is set to 1.")
                slider_batch = gr.Slider(minimum=1, maximum=6, value=4, step=1, label="Batch size", info="The number of images to be sampled.")
                slider_temperature = gr.Slider(minimum=0.1, maximum=2, value=1.4, step=0.1, label="Temperature", info="Control the diversity of layout planner. Higher value indicates more diversity.")
                # slider_seed = gr.Slider(minimum=1, maximum=10000, label="Seed", randomize=True)
                button = gr.Button("Generate")



                guest_id_box = gr.Textbox(label="guest_id", value=f"-1")
                i.select(get_pixels,[i,t,guest_id_box],[i,guest_id_box])
                redo.click(exe_redo, [i,t,guest_id_box],[i])
                undo.click(exe_undo, [i,t,guest_id_box],[i])
                skip_button.click(skip_fun, [i,t,guest_id_box])

                            
            with gr.Column(scale=1):
                output = gr.Gallery(label='Generated image')

                with gr.Accordion("Intermediate results", open=False):
                    gr.Markdown("Composed prompt")
                    composed_prompt = gr.Textbox(label='')
                    gr.Markdown("Layout visualization")
                    layout = gr.Image(height=256, width=256)


        button.click(text_to_image, inputs=[guest_id_box, prompt,keywords,positive_prompt, radio,slider_step,slider_guidance,slider_batch,slider_temperature,slider_natural], outputs=[output, composed_prompt, layout])

        gr.Markdown("## Prompt Examples")
        gr.Examples(
            [
                ["A beautiful city skyline stamp of Shanghai", "", False],
                ["The words 'KFC VIVO50' are inscribed upon the wall in a neon light effect", "KFC/VIVO50", False],
                ["A logo of superman", "", False],
                ["A pencil sketch of a tree with the title nothing to tree here", "", False],
                ["handwritten signature of peter", "", False],
                ["Delicate greeting card of happy birthday to xyz", "", False],
                ["Book cover of good morning baby ", "", False],
                ["The handwritten words Hello World displayed on a wall in a neon light effect", "", False],
                ["Logo of winter in artistic font, made by snowflake", "", False],
                ["A book cover named summer vibe", "", False],
                ["Newspaper with the title Love Story", "", False],
                ["A logo for the company EcoGrow, where the letters look like plants", "EcoGrow", False],
                ["A poster titled 'Quails of North America', showing different kinds of quails.", "Quails/of/North/America", False],
                ["A detailed portrait of a fox guardian with a shield with Kung Fu written on it, by victo ngai and justin gerard, digital art, realistic painting", "kung/fu", False],
                ["A stamp of breath of the wild", "breath/of/the/wild", False],
                ["Poster of the incoming movie Transformers", "Transformers", False],
                ["Some apples are on a table", "", True],
                ["a hotdog with mustard and other toppings on it", "", True],
                ["a bathroom that has a slanted ceiling and a large bath tub", "", True],
                ["a man holding a tennis racquet on a tennis court", "", True],
                ["hamburger with bacon, lettuce, tomato and cheese| promotional image| hyperquality| products shot| full - color| extreme render| mouthwatering", "", True],
            ],
            [
                prompt,
                keywords,
                slider_natural
            ],
            examples_per_page=25
        )

    gr.HTML(
        """
        <div style="text-align: justify; max-width: 1100px; margin: 20px auto;">
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Version</b>: 1.0
        </h3>
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Contact</b>: 
        For help or issues using TextDiffuser-2, please email Jingye Chen <a href="mailto:qwerty.chen@connect.ust.hk">(qwerty.chen@connect.ust.hk)</a>, Yupan Huang <a href="mailto:huangyp28@mail2.sysu.edu.cn">(huangyp28@mail2.sysu.edu.cn)</a> or submit a GitHub issue. For other communications related to TextDiffuser-2, please contact Lei Cui <a href="mailto:lecu@microsoft.com">(lecu@microsoft.com)</a> or Furu Wei <a href="mailto:fuwei@microsoft.com">(fuwei@microsoft.com)</a>.
        </h3>
        <h3 style="font-weight: 450; font-size: 0.8rem; margin: 0rem">
        <b>Disclaimer</b>: 
        Please note that the demo is intended for academic and research purposes <b>ONLY</b>. Any use of the demo for generating inappropriate content is strictly prohibited. The responsibility for any misuse or inappropriate use of the demo lies solely with the users who generated such content, and this demo shall not be held liable for any such use.
        </h3>
        </div>
        """
    )


demo.launch()
