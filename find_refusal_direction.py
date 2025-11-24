# find_refusal_direction.py
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm
import os

class RefusalDirectionFinder:
    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct"):
        print(f"Loading model and processor: {model_name}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16, # Model dtype
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.activations = {}
        
        self.text_decoder_layers = self.model.model.language_model.layers
        self.num_text_layers = len(self.text_decoder_layers)
        print(f"Found {self.num_text_layers} text decoder layers at model.model.language_model.layers")

    def _get_activation(self, name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else: 
                self.activations[name] = output.detach()
        return hook

    def register_hooks(self, target_layer_indices):
        hooks = []
        for layer_idx in target_layer_indices:
            if 0 <= layer_idx < self.num_text_layers:
                layer = self.text_decoder_layers[layer_idx]
                hook = layer.register_forward_hook(self._get_activation(f'layer_{layer_idx}'))
                hooks.append(hook)
            else:
                print(f"Warning: Layer index {layer_idx} is out of bounds for {self.num_text_layers} layers.")
        return hooks

    def get_last_token_activations(self, prompt_text):
        messages = [{"role": "user", "content": prompt_text}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            _ = self.model(**inputs)
        
        last_token_activations = {}
        for name, activation in self.activations.items():
            # Cast to float32 before converting to NumPy for compatibility
            last_token_act = activation[:, -1, :].cpu().float().numpy() # <-- CRITICAL FIX
            last_token_activations[name] = last_token_act
        
        self.activations.clear() 
        return last_token_activations

    def compute_refusal_direction(self, dataset_path, output_path="./refusal_directions_qwen3_vl_2b.pt"):
        dataset = load_from_disk(dataset_path)
        
        print(f"Total text decoder layers available: {self.num_text_layers}")
        target_layer_indices = list(range(self.num_text_layers // 2, self.num_text_layers))
        print(f"Targeting text decoder layer indices: {target_layer_indices}")

        hooks = self.register_hooks(target_layer_indices)

        harmful_acts = {f'layer_{i}': [] for i in target_layer_indices}
        harmless_acts = {f'layer_{i}': [] for i in target_layer_indices}

        print("Extracting activations...")
        for example in tqdm(dataset):
            prompt = example['prompt']
            is_harmful = example['is_harmful']
            
            layer_activations = self.get_last_token_activations(prompt)
            
            target_dict = harmful_acts if is_harmful else harmless_acts
            for layer_name, activation in layer_activations.items():
                target_dict[layer_name].append(activation)
        
        for hook in hooks: hook.remove() 
        print("Activation extraction complete. Computing refusal directions...")

        refusal_directions = {}
        for layer_idx_str in harmful_acts.keys(): 
            actual_layer_idx = int(layer_idx_str.split('_')[1])
            if actual_layer_idx in target_layer_indices:
                harmful_mean = np.vstack(harmful_acts[layer_idx_str]).mean(axis=0)
                harmless_mean = np.vstack(harmless_acts[layer_idx_str]).mean(axis=0)
                
                refusal_dir = harmful_mean - harmless_mean
                refusal_dir = refusal_dir / (np.linalg.norm(refusal_dir) + 1e-8) 
                
                # Save as bfloat16 to match model dtype, or float32. bfloat16 is fine.
                refusal_directions[layer_idx_str] = torch.tensor(refusal_dir, dtype=torch.bfloat16) 
                print(f"Computed direction for {layer_idx_str} (hidden_dim: {refusal_dir.shape[0]}), norm: {np.linalg.norm(refusal_dir):.4f}")
            else:
                print(f"Skipping {layer_idx_str} as it was not a target layer.")

        torch.save(refusal_directions, output_path)
        print(f"Refusal directions saved to {output_path}")
        return refusal_directions, target_layer_indices

if __name__ == "__main__":
    finder = RefusalDirectionFinder()
    if not os.path.exists("./refusal_dataset_qwen3_vl_2b"):
        print("Dataset not found. Please run create_dataset.py first.")
    else:
        refusal_dirs, target_layers = finder.compute_refusal_direction(dataset_path="./refusal_dataset_qwen3_vl_2b")
        print("✅ Refusal direction finding complete!")
        print(f"Targeted layers: {target_layers}")
        print(f"Computed directions for layers: {list(refusal_dirs.keys())}")
