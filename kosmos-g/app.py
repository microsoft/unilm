from app_model import AppModel
from app_utils import *
from controlnet.app_canny import create_demo_canny
from controlnet.app_depth import create_demo_depth
from controlnet.app_ip2p import create_demo_ip2p
from controlnet.app_lineart import create_demo_lineart
from controlnet.app_mlsd import create_demo_mlsd
from controlnet.app_normal import create_demo_normal
from controlnet.app_openpose import create_demo_openpose
from controlnet.app_scribble import create_demo_scribble
from controlnet.app_scribble_interactive import create_demo_scribble_interactive
from controlnet.app_segmentation import create_demo_segmentation
from controlnet.app_shuffle import create_demo_shuffle
from controlnet.app_softedge import create_demo_softedge
from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as distributed_utils


def main(cfg):
    appmodel = AppModel(cfg)

    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column(scale=5):
                ckpt = gr.Textbox(value=cfg.model.pretrained_ckpt_path, show_label=False, container=False)
            with gr.Column(scale=4):
                current_ckpt = gr.Textbox(show_label=False, container=False)
            with gr.Column(scale=1, min_width=100):
                scheduler = gr.Dropdown(['dpms', 'pndm', 'ddim'], value='dpms', show_label=False, container=False,
                                        min_width=60)
        with gr.Row():
            with gr.Column(scale=5):
                lora = gr.Dropdown(['None'] + appmodel.get_available_lora(), value=cfg.model.lora_name,
                                   show_label=False, container=False)
            with gr.Column(scale=4):
                current_lora = gr.Textbox(show_label=False, container=False)
            with gr.Column(scale=1, min_width=60):
                set_ckpt_scheduler_button = gr.Button('Set', container=False, min_width=60)

        set_ckpt_scheduler_button.click(
            fn=appmodel.set_ckpt_scheduler_fn, inputs=[ckpt, scheduler], outputs=current_ckpt, queue=False
        ).then(fn=appmodel.load_lora, inputs=lora, outputs=current_lora, queue=False)

        with gr.Tabs():
            with gr.TabItem('KOSMOS-G'):
                with gr.Blocks():
                    with gr.Row():
                        with gr.Column(scale=1):
                            prompt = gr.Textbox(label="Prompt", max_lines=1,
                                                placeholder="Use <i> to represent the images in prompt")
                            num_input_images = gr.Slider(1, MAX_INPUT_IMAGES, value=DEFAULT_INPUT_IMAGES, step=1,
                                                         label="Number of input images:")
                            input_images = [gr.Image(label=f'img{i}', type="pil",
                                                     visible=True if i < DEFAULT_INPUT_IMAGES else False)
                                            for i in range(MAX_INPUT_IMAGES)]
                            num_input_images.change(variable_images, num_input_images, input_images)

                            seed = gr.Slider(label="Seed", minimum=MIN_SEED, maximum=MAX_SEED, step=1, value=0)
                            randomize_seed = gr.Checkbox(label='Randomize seed', value=False)
                            run_button = gr.Button(label="Run")
                            with gr.Accordion("Advanced options", open=False):
                                lora_scale = gr.Slider(0, 1, value=0, step=0.05, label="LoRA Scale")
                                num_inference_steps = gr.Slider(label="num_inference_steps", minimum=10, maximum=100,
                                                                value=20, step=5)
                                text_guidance_scale = gr.Slider(1, 15, value=7.5, step=0.5, label="Text Guidance Scale")
                                negative_prompt = gr.Textbox(label="Negative Prompt", max_lines=1,
                                                             value="")
                                num_images_per_prompt = gr.Slider(1, MAX_IMAGES_PER_PROMPT,
                                                                  value=4, step=1, label="Number of Images")
                        with gr.Column(scale=2):
                            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery",
                                                        columns=2, height='100%')

                    ips = [prompt, lora_scale, num_inference_steps, text_guidance_scale, negative_prompt,
                           num_images_per_prompt, *input_images]

                    prompt.submit(
                        fn=appmodel.set_ckpt_scheduler_fn, inputs=[ckpt, scheduler], outputs=current_ckpt, queue=False
                    ).then(fn=appmodel.load_lora, inputs=lora, outputs=current_lora, queue=False).then(
                        fn=randomize_seed_fn, inputs=[seed, randomize_seed], outputs=seed, queue=False, api_name=False
                    ).then(fn=appmodel.kosmosg_generation, inputs=ips, outputs=result_gallery)

                    run_button.click(
                        fn=appmodel.set_ckpt_scheduler_fn, inputs=[ckpt, scheduler], outputs=current_ckpt, queue=False
                    ).then(fn=appmodel.load_lora, inputs=lora, outputs=current_lora, queue=False).then(
                        fn=randomize_seed_fn, inputs=[seed, randomize_seed], outputs=seed, queue=False, api_name=False
                    ).then(fn=appmodel.kosmosg_generation, inputs=ips, outputs=result_gallery)

                    gr.Examples(
                        examples=[
                            ['<i>', 'appimg/dog.jpg', None],
                            ['<i> in Minecraft', 'appimg/dog.jpg', None],
                            ['<i> in Batman suit', 'appimg/dog.jpg', None],
                            ['<i> swimming underwater', 'appimg/dog.jpg', None],
                            ['<i> on <i>', 'appimg/dog.jpg', 'appimg/beach.jpg'],
                            ['<i>', 'appimg/bengio.jpg', None],
                            ['<i> as an oil painting in the style of <i>', 'appimg/bengio.jpg', 'appimg/vangogh.jpg'],
                            ['<i> swimming in the pool', 'appimg/bengio.jpg', None],
                            ['<i> wearing <i>', 'appimg/bengio.jpg', 'appimg/sunglasses.jpg'],
                            ['<i> in <i>\'s jacket', 'appimg/bengio.jpg', 'appimg/huang.jpg'],
                            ['<i> plays <i>', 'appimg/astronaut.jpg', 'appimg/guitar.jpg']
                        ],
                        inputs=[prompt, input_images[0], input_images[1]],
                        cache_examples=False,
                        examples_per_page=100
                    )

            with gr.TabItem('ControlNet KOSMOS-G'):
                with gr.Tabs():
                    with gr.TabItem('Canny'):
                        create_demo_canny(appmodel.controlnet_generation_canny)
                    with gr.TabItem('MLSD'):
                        create_demo_mlsd(appmodel.controlnet_generation_mlsd)
                    with gr.TabItem('Scribble'):
                        create_demo_scribble(appmodel.controlnet_generation_scribble)
                    with gr.TabItem('Scribble Interactive'):
                        create_demo_scribble_interactive(appmodel.controlnet_generation_scribble_interactive)
                    with gr.TabItem('SoftEdge'):
                        create_demo_softedge(appmodel.controlnet_generation_softedge)
                    with gr.TabItem('OpenPose'):
                        create_demo_openpose(appmodel.controlnet_generation_openpose)
                    with gr.TabItem('Segmentation'):
                        create_demo_segmentation(appmodel.controlnet_generation_segmentation)
                    with gr.TabItem('Depth'):
                        create_demo_depth(appmodel.controlnet_generation_depth)
                    with gr.TabItem('Normal map'):
                        create_demo_normal(appmodel.controlnet_generation_normal)
                    with gr.TabItem('Lineart'):
                        create_demo_lineart(appmodel.controlnet_generation_lineart)
                    with gr.TabItem('Content Shuffle'):
                        create_demo_shuffle(appmodel.controlnet_generation_shuffle)
                    with gr.TabItem('Instruct Pix2Pix'):
                        create_demo_ip2p(appmodel.controlnet_generation_ip2p)

    app.queue(concurrency_count=1).launch(share=True)


if __name__ == "__main__":
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=None)

    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(cfg, main)
