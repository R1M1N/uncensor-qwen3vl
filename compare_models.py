# compare_models.py

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import argparse

# --- Paste the full, corrected AbliteratedQwen3VLWrapper class here ---
# (This is the class from your apply_and_test_abliteration.py file)
class AbliteratedQwen3VLWrapper:
    def __init__(self, model_name, refusal_directions_path, ablation_strength=1.0):
        print(f"Loading base model for abliteration: {model_name}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device

        print(f"Loading refusal directions from: {refusal_directions_path}")
        self.refusal_directions = torch.load(refusal_directions_path, map_location=self.device)
        
        self.ablation_strength = ablation_strength
        self.hooks = []
        
        self.text_decoder_layers = self.model.model.language_model.layers
        self.num_text_layers = len(self.text_decoder_layers)
        print(f"Identified {self.num_text_layers} text decoder layers for abliteration hook application.")

    def _create_abliteration_hook(self, layer_name, refusal_dir_tensor):
        def hook(module, input, output):
            cache = None
            if isinstance(output, tuple):
                hidden_states = output[0]
                if len(output) > 1:
                    cache = output[1:]
            else:
                hidden_states = output
            
            refusal_dir = refusal_dir_tensor.view(1, 1, -1).to(hidden_states.device)
            projection = (hidden_states * refusal_dir).sum(dim=-1, keepdim=True)
            abliterated_states = hidden_states - (self.ablation_strength * projection * refusal_dir)
            
            if cache is not None:
                return (abliterated_states,) + cache
            else:
                return abliterated_states
        return hook

    def apply_abliteration_hooks(self):
        if self.hooks: self.remove_hooks()
        
        applied_hooks_count = 0
        for layer_name, refusal_dir_tensor in self.refusal_directions.items():
            layer_idx = int(layer_name.split('_')[1])
            
            if 0 <= layer_idx < self.num_text_layers:
                layer_module = self.text_decoder_layers[layer_idx]
                hook = layer_module.register_forward_hook(
                    self._create_abliteration_hook(layer_name, refusal_dir_tensor)
                )
                self.hooks.append(hook)
                applied_hooks_count += 1
            else:
                print(f"Warning: Layer index {layer_idx} from refusal directions is out of bounds for {self.num_text_layers} layers.")
        
        print(f"Applied {applied_hooks_count} abliteration hooks.")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        # print("Removed all abliteration hooks.") # Optional print

    def generate_response(self, prompt_text, image_path=None, max_new_tokens=256):
        messages = [{"role": "user", "content": []}]
        
        if image_path:
            try:
                image = Image.open(image_path).convert('RGB')
                messages[0]["content"].append({"type": "image", "image": image})
            except FileNotFoundError:
                # print(f"Error: Image file not found at {image_path}") # Optional print
                return "Image file not found." # Return error message
            except Exception as e:
                # print(f"Error loading image {image_path}: {e}") # Optional print
                return "Could not load image." # Return error message
        
        messages[0]["content"].append({"type": "text", "text": prompt_text})

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # The processor needs to handle images if present
        # Ensure images are passed as a list of PIL Image objects if they exist
        image_to_process = [Image.open(image_path).convert('RGB')] if image_path and os.path.exists(image_path) else None
        
        inputs = self.processor(
            text=[text], 
            images=image_to_process,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
        
        response_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        assistant_reply = response_text.split("assistant\n")[-1].strip()
        return assistant_reply

    def __enter__(self):
        self.apply_abliteration_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

# --- Main comparison logic ---
def main():
    parser = argparse.ArgumentParser(description="Compare responses from the original and abliterated Qwen3-VL-2B-Instruct models.")
    parser.add_argument("image_path", type=str, help="Path to the image file to describe.")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.", help="The text prompt to accompany the image.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    args = parser.parse_args()

    MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
    REFUSAL_DIRS_PATH = "./refusal_directions_qwen3_vl_2b.pt" # Path from your previous step

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return

    print("="*80)
    print(f"COMPARISON SCRIPT FOR QWEN3-VL-2B-INSTRUCT")
    print(f"Image: {args.image_path}")
    print(f"Prompt: {args.prompt}")
    print("="*80)

    # --- 1. Load Original Model and Processor ---
    print("\n--- Loading Original Model ---")
    try:
        original_model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        original_processor = AutoProcessor.from_pretrained(MODEL_NAME)
        print("Original model loaded successfully.")
    except Exception as e:
        print(f"Error loading original model: {e}")
        return

    # --- 2. Load Abliterated Model Wrapper ---
    print("\n--- Loading Abliterated Model Wrapper ---")
    try:
        # This will load the base model again and apply refusal directions
        abliterated_wrapper = AbliteratedQwen3VLWrapper(MODEL_NAME, REFUSAL_DIRS_PATH, ablation_strength=1.0)
        print("Abliterated model wrapper loaded successfully.")
    except Exception as e:
        print(f"Error loading abliterated model wrapper: {e}")
        return
    
    # --- Helper function for generating responses (to avoid code duplication) ---
    def _generate_description(model_instance, processor_instance, prompt_text, img_path, max_toks):
        messages = [{"role": "user", "content": []}]
        try:
            image = Image.open(img_path).convert('RGB')
            messages[0]["content"].append({"type": "image", "image": image})
        except Exception as e:
            return f"Error processing image for this model: {e}"
        
        messages[0]["content"].append({"type": "text", "text": prompt_text})
        
        text_input = processor_instance.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # The processor expects a list of images if images are part of the input
        image_to_process = [image] if image else None
        
        inputs = processor_instance(
            text=[text_input],
            images=image_to_process,
            return_tensors="pt"
        ).to(model_instance.device)
        
        with torch.no_grad():
            output_ids = model_instance.generate(**inputs, max_new_tokens=max_toks, do_sample=True, temperature=0.7, top_p=0.9)
        
        decoded_text = processor_instance.batch_decode(output_ids, skip_special_tokens=True)[0]
        return decoded_text.split("assistant\n")[-1].strip()

    # --- 3. Generate and Compare Responses ---
    print("\n" + "="*80)
    print("GENERATING RESPONSES...")
    print("="*80)

    print("\n--- Original Model Output ---")
    original_response = _generate_description(original_model, original_processor, args.prompt, args.image_path, args.max_new_tokens)
    print(original_response)

    print("\n" + "-"*80) # Separator

    print("\n--- Abliterated Model Output ---")
    # Use the 'with' statement for the abliterated model to apply hooks
    with abliterated_wrapper:
        # Use the wrapper's own generate_response method
        abliterated_response = abliterated_wrapper.generate_response(args.prompt, image_path=args.image_path, max_new_tokens=args.max_new_tokens)
    print(abliterated_response)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE.")
    print("="*80)

if __name__ == "__main__":
    main()
