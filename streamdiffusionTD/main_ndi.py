import os
import sys
import time
import threading
import numpy as np
import cv2
import NDIlib as ndi
import fire
import json
import argparse
import signal
from multiprocessing import Process, Queue, Manager
from queue import Empty
from typing import Dict, Literal, Optional, List
import torch
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.viewer import receive_images
from utils.wrapper import StreamDiffusionWrapper
from streamdiffusion.image_utils import pil2tensor, pt_to_numpy, numpy_to_pil, postprocess_image

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

import SpoutGL
from OpenGL import GL
import array
from itertools import repeat


from ndi_spout_utils import spout_capture, spout_transmit, ndi_capture, ndi_transmit, select_ndi_source

if not ndi.initialize():
    raise Exception("NDI cannot initialize")

# Debug Output Directory
debug_output_dir = "debug-outputs"
if not os.path.exists(debug_output_dir):
    os.makedirs(debug_output_dir)

def select_input_type():
    print("\n==============================")
    print("Select the input type:")
    print("1. Spout")
    print("2. NDI")
    print("==============================\n")

    choice = input("Enter your choice (1 for Spout, 2 for NDI): ")
    if choice == "1":
        return "spout"
    elif choice == "2":
        return "ndi"
    else:
        raise Exception("Invalid input type selected")



def osc_server(shared_data, ip='127.0.0.1', port=8247):

    def set_negative_prompt_handler(address, *args):
        shared_data["negative_prompt"] = args[0]

    def set_guidance_scale_handler(address, *args):
        shared_data["guidance_scale"] = args[0]

    def set_delta_handler(address, *args):
        shared_data["delta"] = args[0]

    def set_seed_handler(address, *args):
        shared_data["seed"] = args[0]

    def set_t_list_handler(address, *args):
        shared_data["t_list"] = list(args)  # Assuming args contains the t_list values

    def set_prompt_list_handler(address, *args):
        # Assuming args[0] is a string representation of a list
        prompt_list_str = args[0]
        # Convert the string back to a list
        prompt_list = json.loads(prompt_list_str)
        shared_data["prompt_list"] = prompt_list

    def set_seed_list_handler(address, *args):
        # Assuming args[0] is a string representation of a list of lists
        seed_list_str = args[0]
        # Convert the string back to a list of lists
        seed_list = json.loads(seed_list_str)
        # Convert each inner list to a list with an int and a float
        seed_list = [[int(seed_val), float(weight)] for seed_val, weight in seed_list]
        shared_data["seed_list"] = seed_list
        # print(seed_list)


    def set_sdmode_handler(address, *args):
        shared_data["sdmode"] = args[0]


    def stop_stream_handler(address, *args):
        print("Stop command received. Stopping the stream.")
        shared_data["stop_stream"] = True

    dispatcher = Dispatcher()


    dispatcher.map("/negative_prompt", set_negative_prompt_handler)
    dispatcher.map("/guidance_scale", set_guidance_scale_handler)
    dispatcher.map("/delta", set_delta_handler)
    dispatcher.map("/seed", set_seed_handler)
    dispatcher.map("/t_list", set_t_list_handler)

    dispatcher.map("/prompt_list", set_prompt_list_handler)
    dispatcher.map("/seed_list", set_seed_list_handler)

    dispatcher.map("/sdmode", set_sdmode_handler)

    dispatcher.map("/stop", stop_stream_handler)


    server = BlockingOSCUDPServer((ip, port), dispatcher)
    print(f"Starting OSC server on {ip}:{port}")
    server.serve_forever()  # This will block and run the server

def terminate_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()

def image_generation_process(
    capture_queue: Queue,
    transmit_queue: Queue, 
    model_id_or_path: str,
    lora_dict: Optional[Dict[str, float]],
    prompt: str,
    negative_prompt: str,
    frame_buffer_size: int,
    width: int,
    height: int,
    acceleration: Literal["none", "xformers", "tensorrt"],
    use_denoising_batch: bool,
    seed: int,
    cfg_type: Literal["none", "full", "self", "initialize"],
    guidance_scale: float,
    delta: float,
    do_add_noise: bool,
    enable_similar_image_filter: bool,
    similar_image_filter_threshold: float,
    similar_image_filter_max_skip_frame: float,
    shared_data,  # Make sure to include this parameter
    t_index_list: List[int] ,
    mode:str,
    lcm_lora_id: Optional[str] = None,
    vae_id: Optional[str] = None,


) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    model_id_or_path : str
        The name of the model to use for image generation.
    lora_dict : Optional[Dict[str, float]], optional
        The lora_dict to load, by default None.
        Keys are the LoRA names and values are the LoRA scales.
        Example: {"LoRA_1" : 0.5 , "LoRA_2" : 0.7 ,...}
    prompt : str
        The prompt to generate images from.
    negative_prompt : str, optional
        The negative prompt to use.
    frame_buffer_size : int, optional
        The frame buffer size for denoising batch, by default 1.
    width : int, optional
        The width of the image, by default 512.
    height : int, optional
        The height of the image, by default 512.
    acceleration : Literal["none", "xformers", "tensorrt"], optional
        The acceleration method, by default "tensorrt".
    use_denoising_batch : bool, optional
        Whether to use denoising batch or not, by default True.
    seed : int, optional
        The seed, by default 2. if -1, use random seed.
    cfg_type : Literal["none", "full", "self", "initialize"],
    optional
        The cfg_type for img2img mode, by default "self".
        You cannot use anything other than "none" for txt2img mode.
    guidance_scale : float, optional
        The CFG scale, by default 1.2.
    delta : float, optional
        The delta multiplier of virtual residual noise,
        by default 1.0.
    do_add_noise : bool, optional
        Whether to add noise for following denoising steps or not,
        by default True.
    enable_similar_image_filter : bool, optional
        Whether to enable similar image filter or not,
        by default False.
    similar_image_filter_threshold : float, optional
        The threshold for similar image filter, by default 0.98.
    similar_image_filter_max_skip_frame : int, optional
        The max skip frame for similar image filter, by default 10.
    """

    global inputs
    global stop_capture
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        lora_dict=lora_dict,
        t_index_list=t_index_list,
        frame_buffer_size=frame_buffer_size,
        width=width,
        height=height,
        warmup=10,
        acceleration=acceleration,
        do_add_noise=do_add_noise,
        enable_similar_image_filter=enable_similar_image_filter,
        similar_image_filter_threshold=similar_image_filter_threshold,
        similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame,
        use_denoising_batch=use_denoising_batch,
        cfg_type=cfg_type,
        seed=seed,
        lcm_lora_id=lcm_lora_id,  # Add this line
        vae_id=vae_id,            # And this line

    )

    current_prompt = prompt
    current_prompt_list = shared_data.get("prompt_list", [[prompt, 1.0]])
    current_seed_list = shared_data.get("seed_list", [[seed, 1.0]])
    
    noise_bank = {}
    prompt_cache = {}

    print('Preparing Stream...')

    stream.prepare(
        prompt=current_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
    )

    time.sleep(1)

    start_time = time.time()
    prompt_changed = False
    frame_count = 0

    while True:
        try:
            if mode == "img2img" and not capture_queue.empty():
                frame_np = capture_queue.get(block=False)

                if frame_np.shape[2] == 3:
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)

                input_tensor = to_tensor(frame_np).to(dtype=torch.float32)
                processed_tensor = stream.stream(input_tensor)

                processed_np = postprocess_image(processed_tensor, output_type="np")
                processed_np = (processed_np * 255).astype(np.uint8)                

                transmit_queue.put(processed_np)

            elif mode == "txt2img":
                try:
                    # processed_np = stream.txt2img()  # Attempt to generate image from text
                    processed_np = custom_txt2img_using_prepared_noise(stream_diffusion=stream.stream, expected_batch_size=1, output_type='np')
                    if processed_np.max() <= 1.0:
                        processed_np = (processed_np * 255).astype(np.uint8)
                    transmit_queue.put(processed_np)
                except Exception as e:
                    print(f"Exception during txt2img processing: {str(e)}")

            frame_count += 1

            new_sdmode = shared_data.get("sdmode", mode)
            if new_sdmode != mode:
                mode = new_sdmode

            # PROMPT DICT + GUIDANCE SCALE + DELTA
            new_guidance_scale = float(shared_data.get("guidance_scale", guidance_scale))
            new_delta = float(shared_data.get("delta", delta))
            new_prompt_list = shared_data.get("prompt_list", {})
            new_negative_prompt = shared_data.get("negative_prompt", negative_prompt)
            # Check if there is an actual change in parameters
            if (new_prompt_list != current_prompt_list or 
                new_guidance_scale != guidance_scale or 
                new_delta != delta or 
                new_negative_prompt != negative_prompt):

                # Update the current values
                current_prompt_list = new_prompt_list
                guidance_scale = new_guidance_scale
                delta = new_delta
                negative_prompt = new_negative_prompt

                update_combined_prompts_and_parameters(
                    stream.stream, 
                    current_prompt_list, 
                    guidance_scale, 
                    delta, 
                    negative_prompt,
                    prompt_cache
                )

            ##SEED DICT
            new_seed_list = shared_data.get("seed_list", current_seed_list)
            if new_seed_list != current_seed_list:
                current_seed_list = new_seed_list
                # Check if all weights are zero
                if any(weight > 0 for _, weight in current_seed_list):
                    blended_noise = blend_noise_tensors(current_seed_list, noise_bank, stream.stream)
                    stream.stream.init_noise = blended_noise

            ##T_LIST
            new_t_list = shared_data.get("t_list", t_index_list)
            if new_t_list != stream.stream.t_list:
                update_t_list_attributes(stream.stream, new_t_list)

            ##STOP STREAM
            if shared_data.get("stop_stream", False):
                print("Stopping image generation process.")
                break

            else:
                time.sleep(0.005)
        except KeyboardInterrupt:
            break

# function to update t_list-related attributes
def update_t_list_attributes(stream_diffusion_instance, new_t_list):
    stream_diffusion_instance.t_list = new_t_list
    stream_diffusion_instance.sub_timesteps = [stream_diffusion_instance.timesteps[t] for t in new_t_list]
    
    sub_timesteps_tensor = torch.tensor(
        stream_diffusion_instance.sub_timesteps, dtype=torch.long, device=stream_diffusion_instance.device
    )
    stream_diffusion_instance.sub_timesteps_tensor = torch.repeat_interleave(
        sub_timesteps_tensor, 
        repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, 
        dim=0
    )

    c_skip_list = []
    c_out_list = []
    for timestep in stream_diffusion_instance.sub_timesteps:
        c_skip, c_out = stream_diffusion_instance.scheduler.get_scalings_for_boundary_condition_discrete(timestep)
        c_skip_list.append(c_skip)
        c_out_list.append(c_out)

    stream_diffusion_instance.c_skip = torch.stack(c_skip_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    stream_diffusion_instance.c_out = torch.stack(c_out_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)

    # Calculate alpha_prod_t_sqrt and beta_prod_t_sqrt
    alpha_prod_t_sqrt_list = []
    beta_prod_t_sqrt_list = []
    for timestep in stream_diffusion_instance.sub_timesteps:
        alpha_prod_t_sqrt = stream_diffusion_instance.scheduler.alphas_cumprod[timestep].sqrt()
        beta_prod_t_sqrt = (1 - stream_diffusion_instance.scheduler.alphas_cumprod[timestep]).sqrt()
        alpha_prod_t_sqrt_list.append(alpha_prod_t_sqrt)
        beta_prod_t_sqrt_list.append(beta_prod_t_sqrt)

    alpha_prod_t_sqrt = torch.stack(alpha_prod_t_sqrt_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)
    beta_prod_t_sqrt = torch.stack(beta_prod_t_sqrt_list).view(len(new_t_list), 1, 1, 1).to(dtype=stream_diffusion_instance.dtype, device=stream_diffusion_instance.device)

    stream_diffusion_instance.alpha_prod_t_sqrt = torch.repeat_interleave(alpha_prod_t_sqrt, repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, dim=0)
    stream_diffusion_instance.beta_prod_t_sqrt = torch.repeat_interleave(beta_prod_t_sqrt, repeats=stream_diffusion_instance.frame_bff_size if stream_diffusion_instance.use_denoising_batch else 1, dim=0)

@torch.no_grad()
def update_combined_prompts_and_parameters(stream_diffusion, prompt_list, new_guidance_scale, new_delta, new_negative_prompt, prompt_cache):
    # Update guidance_scale and delta
    stream_diffusion.guidance_scale = new_guidance_scale
    stream_diffusion.delta = new_delta

    # Update stock_noise if necessary
    if stream_diffusion.guidance_scale > 1.0 and (stream_diffusion.cfg_type in ["self", "initialize"]):
        stream_diffusion.stock_noise *= stream_diffusion.delta

    # Initialize combined embeddings
    combined_embeds = None

    # Current prompts to keep in the cache
    current_prompts = set()

    for idx, prompt in enumerate(prompt_list):
        prompt_text, weight = prompt  # Unpack the list
        if weight == 0:
            continue

        current_prompts.add(idx)

        # Check if prompt is already encoded and in the cache
        if idx not in prompt_cache or prompt_cache[idx]['text'] != prompt_text:
            # Encode new prompt and add to cache
            encoder_output = stream_diffusion.pipe.encode_prompt(
                prompt=prompt_text,
                device=stream_diffusion.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=stream_diffusion.guidance_scale > 1.0,
                negative_prompt=new_negative_prompt,
            )
            # Store both the encoded prompt and the original text in the cache
            prompt_cache[idx] = {'embed': encoder_output[0], 'text': prompt_text}

        # Apply weight to the prompt embeddings (from cache)
        weighted_embeds = prompt_cache[idx]['embed'] * weight

        # Combine embeddings
        if combined_embeds is None:
            combined_embeds = weighted_embeds
        else:
            combined_embeds += weighted_embeds

    # Update prompt embeddings if valid combined embeddings are present
    if combined_embeds is not None:
        stream_diffusion.prompt_embeds = combined_embeds.repeat(stream_diffusion.batch_size, 1, 1)

    # Clear unused prompts from the cache
    unused_prompts = set(prompt_cache.keys()) - current_prompts
    for prompt in unused_prompts:
        del prompt_cache[prompt]

# LINEAR SEED INTERPOLATION
def blend_noise_tensors(seed_list, noise_bank, stream_diffusion):
    blended_noise = None
    total_weight = 0

    for seed_val, weight in seed_list:  # Unpack the list
        if weight == 0:
            continue

        noise_tensor = noise_bank.get(seed_val)
        if noise_tensor is None:
            # Generate noise tensor for this seed if not already done
            generator = torch.Generator().manual_seed(seed_val)
            noise_tensor = torch.randn(
                (stream_diffusion.batch_size, 4, stream_diffusion.latent_height, stream_diffusion.latent_width),
                generator=generator
            ).to(device=stream_diffusion.device, dtype=stream_diffusion.dtype)
            noise_bank[seed_val] = noise_tensor

        if blended_noise is None:
            blended_noise = noise_tensor * weight
        else:
            blended_noise += noise_tensor * weight
        total_weight += weight

    return blended_noise

def custom_txt2img_using_prepared_noise(stream_diffusion, expected_batch_size, output_type='np'):
    # If the current batch size is larger than expected, use only the first `expected_batch_size` entries
    if stream_diffusion.init_noise.size(0) > expected_batch_size:
        adjusted_noise = stream_diffusion.init_noise[:expected_batch_size]
    elif stream_diffusion.init_noise.size(0) < expected_batch_size:
        # If you need to increase the batch size, repeat the tensor
        repeats = [expected_batch_size // stream_diffusion.init_noise.size(0)] + [-1] * (stream_diffusion.init_noise.dim() - 1)
        adjusted_noise = stream_diffusion.init_noise.repeat(*repeats)[:expected_batch_size]
    else:
        adjusted_noise = stream_diffusion.init_noise

    x_0_pred_out = stream_diffusion.predict_x0_batch(adjusted_noise)
    x_output = stream_diffusion.decode_image(x_0_pred_out).detach().clone()

    if output_type == 'np':
        x_output = postprocess_image(x_output, output_type=output_type)

    return x_output


def main():
    def signal_handler(sig, frame):
        print('Exiting...')
        terminate_processes([capture_process, osc_process, generation_process, transmit_process])
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
        
    parser = argparse.ArgumentParser(description="StreamDiffusion NDI Stream Script")
    parser.add_argument('-c', '--config', type=str, default='stream_config.json', help='Path to the configuration file')
    parser.add_argument('--input', type=str, choices=['spout', 'ndi'], help='Input type: spout or ndi')
    parser.add_argument('--sender', type=str, help='Sender name for Spout or NDI')

    args = parser.parse_args()

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(current_script_dir, args.config)
    print(f"Config file path: {config_file_path}")
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)
    # Open and read the config file
    with open(config_file_path, 'r') as config_file:
        config = json.load(config_file)

    print("\n====================================")
    print("StreamDiffusion: Starting the NDI capture and transmit script...")
    print("====================================\n")
    osc_receive_port = config.get("osc_out_port", 8247)
    osc_transmit_port = config.get("osc_in_port", 8248)
    model_id_or_path = config["model_id_or_path"]
    lora_dict = config["lora_dict"]
    prompt = config["prompt"]
    negative_prompt = config["negative_prompt"]
    frame_buffer_size = config["frame_buffer_size"]
    width = config["width"]
    height = config["height"]
    acceleration = config["acceleration"]
    use_denoising_batch = config["use_denoising_batch"]
    seed = config["seed"]
    cfg_type = config["cfg_type"]
    guidance_scale = config["guidance_scale"]
    delta = config["delta"]
    do_add_noise = config["do_add_noise"]
    enable_similar_image_filter = config["enable_similar_image_filter"]
    similar_image_filter_threshold = config["similar_image_filter_threshold"]
    similar_image_filter_max_skip_frame = config["similar_image_filter_max_skip_frame"]
    t_index_list = config.get("t_index_list", [25, 40])
    mode=config.get("sdmode","img2img")
    lcm_lora_id = config.get("lcm_lora_id")
    vae_id = config.get("vae_id")

    print(f"Model ID or Path: {model_id_or_path}\n")
    if lcm_lora_id is not None:
        print(f"LCM LoRA ID: {lcm_lora_id}\n")
    else:
        print("LCM LoRA ID: None\n")
    if vae_id is not None:
        print(f"VAE ID: {vae_id}\n")
    else:
        print("VAE ID: None\n")
    if lora_dict is not None:
        for model_path, weight in lora_dict.items():
            print(f"LoRA Model: {model_path}, Weight: {weight}\n\n")
    print("====================================\n")


    if os.path.isfile(model_id_or_path):
        model_id_or_path = model_id_or_path.replace('/', '\\')
    if args.input:
        input_type = args.input
    else:
        input_type = select_input_type()

    capture_queue = Queue()
    transmit_queue = Queue()

    if input_type == "spout":
        spout_sender_name = args.sender if args.sender else "spout_sd_out"
        print("\n====================================")
        print(f"Using Spout sender name: {spout_sender_name}")
        print("====================================\n")
        capture_process = Process(target=spout_capture, args=(capture_queue, spout_sender_name))
    elif input_type == "ndi":
        ndi_sender_name = select_ndi_source(args.sender)  # Pass args.sender to the function
        capture_process = Process(target=ndi_capture, args=(ndi_sender_name, capture_queue))
    capture_process.start()

    with Manager() as manager:
        shared_data = manager.dict()
        shared_data["prompt"] = prompt

        osc_process = Process(target=osc_server, args=(shared_data, '127.0.0.1', osc_receive_port))
        osc_process.start()

        generation_process = Process(
            target=image_generation_process,
            args=(
                capture_queue,
                transmit_queue,
                model_id_or_path,
                lora_dict,
                prompt,
                negative_prompt,
                frame_buffer_size,
                width,
                height,
                acceleration,
                use_denoising_batch,
                seed,
                cfg_type,
                guidance_scale,
                delta,
                do_add_noise,
                enable_similar_image_filter,
                similar_image_filter_threshold,
                similar_image_filter_max_skip_frame,
                shared_data,
                t_index_list,
                mode,
                lcm_lora_id,  # Added this line
                vae_id,       # And this line
                ),
        )
        generation_process.start()

        print("\nStarting the transmit process...\n")
        if input_type == "spout":
            spout_out_name = 'StreamDiffusion_Spout_out'
            transmit_process = Process(target=spout_transmit, args=(transmit_queue, spout_out_name, osc_transmit_port))
        elif input_type == "ndi":
            transmit_process = Process(target=ndi_transmit, args=(transmit_queue, osc_transmit_port))
        transmit_process.start()

        try:
            while True:
                if shared_data.get("stop_stream", False):
                    print("Stop command received. Initiating shutdown...")
                    break
                time.sleep(0.1)  # Short sleep to prevent high CPU usage

        except KeyboardInterrupt:
            print("KeyboardInterrupt received, signalling to stop subprocesses...")
            shared_data["stop_stream"] = True

        finally:
            terminate_processes([capture_process, osc_process, generation_process, transmit_process])
            capture_process.join()
            generation_process.join()
            transmit_process.join()
            osc_process.join()
            print("All subprocesses terminated. Exiting main process...")
            sys.exit(0)
if __name__ == "__main__":
    fire.Fire(main)