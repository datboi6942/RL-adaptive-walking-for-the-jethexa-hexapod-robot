import torch
import sys
import os

# Add Fast-SCNN repo to path
fast_scnn_repo_path = './Fast-SCNN-pytorch'
if os.path.isdir(fast_scnn_repo_path):
    sys.path.insert(0, fast_scnn_repo_path)
    print(f"Added '{fast_scnn_repo_path}' to sys.path")
else:
    print(f"Error: Directory '{fast_scnn_repo_path}' not found in current directory.")
    sys.exit(1)

try:
    from models.fast_scnn import FastSCNN
    print("Successfully imported FastSCNN from models.fast_scnn")
except ImportError as e:
    print(f"Error importing FastSCNN: {e}")
    print("Please ensure 'Fast-SCNN-pytorch' contains 'models/fast_scnn.py'")
    sys.exit(1)

# Define model path
model_path = './jethexa_gym_env/models/fast_scnn_cityscapes.pth'
print(f'\n--- Loading Checkpoint: {model_path} ---')

if not os.path.isfile(model_path):
    print(f"Error: Checkpoint file not found at '{model_path}'")
    sys.exit(1)

try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print("Checkpoint loaded successfully.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    sys.exit(1)

# Inspect checkpoint structure
print('\n--- Checkpoint Keys (Top Level) ---')
if isinstance(checkpoint, dict):
    print(list(checkpoint.keys()))
    # Try to find the state dictionary
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Using 'state_dict' from checkpoint.")
    elif 'model_state' in checkpoint: # common alternative name
         state_dict = checkpoint['model_state']
         print("Using 'model_state' from checkpoint.")
    else:
         # Assume the whole checkpoint *is* the state_dict if no common keys found
         state_dict = checkpoint
         print("Assuming the entire checkpoint is the state_dict.")
else:
    # If checkpoint is not a dict, assume it's the state_dict directly
    state_dict = checkpoint
    print('Checkpoint is not a dictionary, assuming it IS the state_dict.')

# Print state_dict keys
print('\n--- State Dict Keys (First 10) ---')
if hasattr(state_dict, 'keys'):
    state_dict_keys = list(state_dict.keys())
    print(state_dict_keys[:10])
else:
    print('Error: Loaded state_dict does not have keys (unexpected format).')
    sys.exit(1)

# Instantiate the model and print its expected keys
print('\n--- Instantiating Model ---')
try:
    model = FastSCNN(num_classes=19)
    print("Model instantiated successfully.")
except Exception as e:
    print(f"Error instantiating FastSCNN model: {e}")
    sys.exit(1)

print('\n--- Model Expected Keys (First 10) ---')
model_keys = list(model.state_dict().keys())
print(model_keys[:10])

# Basic comparison
print('\n--- Comparison ---')
if not state_dict_keys:
    print("Cannot compare keys: State dict keys list is empty.")
elif not model_keys:
    print("Cannot compare keys: Model keys list is empty.")
elif state_dict_keys[0] == model_keys[0]:
     print("First key matches.")
else:
     print(f"First key MISMATCH: Checkpoint has '{state_dict_keys[0]}', Model expects '{model_keys[0]}'")

if state_dict_keys and model_keys and state_dict_keys[0].startswith('module.') and not model_keys[0].startswith('module.'):
    print("Hint: Checkpoint keys might have a 'module.' prefix (from DataParallel saving).") 