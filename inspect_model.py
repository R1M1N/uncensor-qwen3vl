# inspect_model.py
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

print(f"Loading model: {MODEL_NAME}")
# Load with low memory to inspect if needed, or just load normally
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16, # or torch.float16
    device_map="cpu", # Load on CPU for inspection to save GPU VRAM
    trust_remote_code=True
)

print("\n--- Model Architecture ---")
print(model)

print("\n--- Model Named Children (first level) ---")
for name, module in model.named_children():
    print(f"{name}: {module}")

print("\n--- Inspecting 'model' attribute (if it exists) ---")
if hasattr(model, 'model'):
    print("Type of model.model:", type(model.model))
    if hasattr(model.model, 'named_children'):
        for name, module in model.model.named_children():
            print(f"  model.model.{name}: {type(module)}")
            # If we find layers here, we can dig deeper
            # For example, if it's 'blocks' or 'h' or 'layers'
            # if hasattr(module, 'named_children'):
            #     for sub_name, sub_module in module.named_children():
            #         print(f"    model.model.{name}.{sub_name}: {type(sub_module)}")

# A more direct way to find transformer blocks/layers
print("\n--- Searching for common layer/block names ---")
all_named_modules = dict(model.named_modules())
potential_layer_paths = []
for name, module in all_named_modules.items():
    if 'layers' in name or 'blocks' in name or 'h.' in name: # 'h.' is common in GPT-2 like models
        # Check if it's a ModuleList or a list of transformer blocks
        if isinstance(module, torch.nn.ModuleList) or (hasattr(module, '__len__') and len(module) > 0 and hasattr(module[0], 'self_attn')):
             potential_layer_paths.append(name)
             print(f"Potential layer path: {name} (Type: {type(module)}, Length: {len(module)})")

if not potential_layer_paths:
    print("Could not automatically find layer paths. Please inspect the model architecture printed above.")
    print("Look for ModuleList entries that seem like transformer blocks.")
else:
    print(f"\nFound potential layer paths: {potential_layer_paths}")
    # Let's try to get the actual layer objects from one of these paths
    # This will help confirm the correct attribute name
    for path in potential_layer_paths:
        try:
            # Example: if path is "model.layers"
            # layers = getattr(model, "model").layers
            # Example: if path is "model.h"
            # layers = getattr(model, "model").h
            
            # A more generic way to access nested attributes
            current_module = model
            parts = path.split('.')
            for part in parts:
                current_module = getattr(current_module, part)
            
            if isinstance(current_module, torch.nn.ModuleList) and len(current_module) > 0:
                print(f"Successfully accessed layers at '{path}'. Number of layers: {len(current_module)}")
                # Let's inspect the first layer to understand its structure
                print(f"First layer at '{path}[0]': {current_module[0]}")
                # We are looking for attributes like 'self_attn', 'mlp', etc.
                # These are typical components of a transformer block.
                # The hook will be applied to this block.
                break 
        except AttributeError as e:
            print(f"Could not access layers at '{path}': {e}")
