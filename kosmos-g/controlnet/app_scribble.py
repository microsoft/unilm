from app_utils import *


def create_demo_scribble(generation_fn):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Control image")
                prompt = gr.Textbox(label="Prompt", max_lines=1,
                                    placeholder="Use <i> to represent the images in prompt")
                num_input_images = gr.Slider(1, MAX_INPUT_IMAGES, value=DEFAULT_INPUT_IMAGES, step=1,
                                             label="Number of input images:")
                input_images = [
                    gr.Image(label=f'img{i}', type="pil", visible=True if i < DEFAULT_INPUT_IMAGES else False)
                    for i in range(MAX_INPUT_IMAGES)]
                num_input_images.change(variable_images, num_input_images, input_images)

                seed = gr.Slider(label="Seed", minimum=MIN_SEED, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label='Randomize seed', value=False)
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    num_inference_steps = gr.Slider(label="num_inference_steps", minimum=10, maximum=100, value=20,
                                                    step=5)
                    text_guidance_scale = gr.Slider(1, 15, value=7.5, step=0.5, label="Text Guidance Scale")
                    negative_prompt = gr.Textbox(label="Negative Prompt", max_lines=1,
                                                 value="")
                    num_images_per_prompt = gr.Slider(1, MAX_IMAGES_PER_PROMPT, value=DEFAULT_IMAGES_PER_PROMPT, step=1,
                                                      label="Number of Images")
                    image_resolution = gr.Slider(label='Image resolution', minimum=MIN_IMAGE_RESOLUTION,
                                                 maximum=MAX_IMAGE_RESOLUTION, value=DEFAULT_IMAGE_RESOLUTION, step=256)
                    preprocess_resolution = gr.Slider(label='Preprocess resolution', minimum=128, maximum=512,
                                                      value=512, step=1)
                    preprocessor_name = gr.Radio(
                        label='Preprocessor',
                        choices=['HED', 'PidiNet', 'None'],
                        type='value',
                        value='HED')

            with gr.Column(scale=2):
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", columns=2,
                                            height='100%')
        ips = [prompt, num_inference_steps, text_guidance_scale, negative_prompt, num_images_per_prompt, image,
               image_resolution, preprocess_resolution, preprocessor_name, *input_images]

        prompt.submit(
            fn=randomize_seed_fn, inputs=[seed, randomize_seed], outputs=seed, queue=False, api_name=False
        ).then(fn=generation_fn, inputs=ips, outputs=result_gallery)

        run_button.click(
            fn=randomize_seed_fn, inputs=[seed, randomize_seed], outputs=seed, queue=False, api_name=False
        ).then(fn=generation_fn, inputs=ips, outputs=result_gallery)

        gr.Examples(
            examples=controlnet_example,
            inputs=[image, prompt, input_images[0], input_images[1]],
            cache_examples=False,
            examples_per_page=100
        )

    return demo
