# BronchoGAN Usage Guide

Complete guide for generating realistic bronchoscopy images from input images using BronchoGAN.

## Overview

BronchoGAN works in 2 steps:
1. **Step 1**: Convert input image → Depth map (using Depth Anything V2)
2. **Step 2**: Convert depth map → Realistic bronchoscopy image (using pix2pixHD GAN)

---

## Prerequisites

### 1. Install Dependencies
```bash
pip install torch torchvision opencv-python numpy tqdm dominate
```

### 2. Download Model Weights

You need two models:

**a) Depth Anything V2 Model (335M parameters)**
- Download: [depth_anything_v2_vitl.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth)
- Place in: `/home/ahmad/BronchoGAN/Depth-Anything-V2/depth_anything_v2_vitl.pth`

**b) BronchoGAN Generator Model**
- Already in: `/home/ahmad/BronchoGAN/Models/latest_net_G.pth` ✓

---

## Step-by-Step Usage

### Step 1: Convert Image to Depth Map

**Input**: Any bronchoscopy image (`.jpg`, `.png`)
**Output**: Depth map (colored visualization + numpy arrays)

#### Using Python Code:

```python
import cv2
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/ahmad/BronchoGAN/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# Setup device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration
model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# Load model
encoder = 'vitl'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load('/home/ahmad/BronchoGAN/Depth-Anything-V2/depth_anything_v2_vitl.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

# Process single image
def extract_depth(image_path, output_path):
    # Read image (force color mode)
    raw_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Infer depth
    depth = model.infer_image(raw_img)  # HxW raw depth map

    # Inversion for bronchoscopy (higher = closer)
    max_depth = depth.max()
    inverted_depth = max_depth - depth

    # Save numpy arrays
    np.save(output_path.replace('.png', '_depth.npy'), depth)
    np.save(output_path.replace('.png', '_inverted_depth.npy'), inverted_depth)

    # Create colored visualization (VIRIDIS colormap)
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_vis = depth_vis.astype(np.uint8)  # Convert to uint8
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(output_path, depth_colored)

    print(f"✓ Depth saved: {output_path}")
    return depth, inverted_depth

# Example usage
depth_map, inverted = extract_depth(
    '/home/ahmad/BronchoGAN/example.png',
    '/home/ahmad/BronchoGAN/example_depth.png'
)
```

**Outputs:**
- `example_depth.png` - Colored depth visualization (purple=far, yellow=close)
- `example_depth.npy` - Raw depth values (numpy array)
- `example_inverted_depth.npy` - Inverted depth values

---

### Step 2: Generate Realistic Bronchoscopy from Depth

**Input**: Depth map image from Step 1
**Output**: Synthetic realistic bronchoscopy image

#### Method 1: Using Command Line

```bash
cd /home/ahmad/BronchoGAN/pix2pixHD

# Create test data folder
mkdir -p test_data/test_A

# Copy depth image
cp /home/ahmad/BronchoGAN/example_depth.png test_data/test_A/

# Run generation
python test.py \
  --label_nc 0 \
  --no_instance \
  --model 'pix2pixHD' \
  --name latest \
  --checkpoints_dir '../Models' \
  --netG 'global' \
  --dataroot './test_data/' \
  --batchSize 1 \
  --display_winsize 156 \
  --how_many 1 \
  --n_blocks_local 2 \
  --n_blocks_global 2 \
  --n_local_enhancers 2 \
  --loadSize 128 \
  --results_dir './results'
```

**Output Location:**
```
./results/latest/test_latest/images/
├── example_depth_synthesized_image.png   # Your generated bronchoscopy!
└── example_depth_input_label.png         # Input depth (for reference)
```

#### Method 2: Using Python Code

```python
import sys
sys.path.insert(0, '/home/ahmad/BronchoGAN/pix2pixHD')

import torch
import cv2
import numpy as np
from options.test_options import TestOptions
from models.models import create_model
import util.util as util

# Setup options
opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.label_nc = 0
opt.no_instance = True
opt.netG = 'global'
opt.n_blocks_local = 2
opt.n_blocks_global = 2
opt.n_local_enhancers = 2
opt.checkpoints_dir = '/home/ahmad/BronchoGAN/Models'
opt.name = 'latest'
opt.which_epoch = 'latest'

# Load model
print("Loading BronchoGAN model...")
model = create_model(opt)
print("✓ Model loaded!")

# Load depth image
depth_img = cv2.imread('/home/ahmad/BronchoGAN/example_depth.png')
depth_img = cv2.resize(depth_img, (128, 128))  # Resize to model input size

# Convert to tensor
depth_tensor = torch.from_numpy(depth_img).permute(2, 0, 1).unsqueeze(0).float()
depth_tensor = (depth_tensor / 255.0) * 2 - 1  # Normalize to [-1, 1]

# Create dummy instance map (not used)
inst_tensor = torch.zeros(1, 1, 128, 128)

# Run inference
print("Generating bronchoscopy image...")
with torch.no_grad():
    fake_image, segmentation_mask = model.inference(depth_tensor, inst_tensor, None)

# Save result
output = util.tensor2im(fake_image.data[0])
cv2.imwrite('/home/ahmad/BronchoGAN/generated_broncho.png', output)
print("✓ Generated image saved: generated_broncho.png")
```

---

## Complete Pipeline Example

**From original image → depth → synthetic bronchoscopy**

```python
import cv2
import torch
import numpy as np
import sys

# Add paths
sys.path.insert(0, '/home/ahmad/BronchoGAN/Depth-Anything-V2')
sys.path.insert(0, '/home/ahmad/BronchoGAN/pix2pixHD')

from depth_anything_v2.dpt import DepthAnythingV2
from options.test_options import TestOptions
from models.models import create_model
import util.util as util

# ==== STEP 1: Load Depth Model ====
print("Step 1: Loading Depth Anything V2...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_configs = {'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}}
depth_model = DepthAnythingV2(**model_configs['vitl'])
depth_model.load_state_dict(torch.load('/home/ahmad/BronchoGAN/Depth-Anything-V2/depth_anything_v2_vitl.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()
print("✓ Depth model loaded")

# ==== STEP 2: Extract Depth ====
print("\nStep 2: Extracting depth from image...")
input_image_path = '/home/ahmad/BronchoGAN/example.png'
raw_img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
depth = depth_model.infer_image(raw_img)

# Create colored depth visualization
depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255
depth_vis = depth_vis.astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
cv2.imwrite('/home/ahmad/BronchoGAN/depth_output.png', depth_colored)
print("✓ Depth extracted and saved")

# ==== STEP 3: Load GAN Model ====
print("\nStep 3: Loading BronchoGAN model...")
opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.label_nc = 0
opt.no_instance = True
opt.netG = 'global'
opt.n_blocks_local = 2
opt.n_blocks_global = 2
opt.n_local_enhancers = 2
opt.checkpoints_dir = '/home/ahmad/BronchoGAN/Models'
opt.name = 'latest'
gan_model = create_model(opt)
print("✓ GAN model loaded")

# ==== STEP 4: Generate Bronchoscopy ====
print("\nStep 4: Generating synthetic bronchoscopy...")
depth_img = cv2.resize(depth_colored, (128, 128))
depth_tensor = torch.from_numpy(depth_img).permute(2, 0, 1).unsqueeze(0).float()
depth_tensor = (depth_tensor / 255.0) * 2 - 1
inst_tensor = torch.zeros(1, 1, 128, 128)

with torch.no_grad():
    fake_image, segmentation_mask = gan_model.inference(depth_tensor, inst_tensor, None)

output = util.tensor2im(fake_image.data[0])
cv2.imwrite('/home/ahmad/BronchoGAN/final_broncho.png', output)
print("✓ Synthetic bronchoscopy saved: final_broncho.png")
print("\n✓✓✓ COMPLETE! ✓✓✓")
```

---

## Important Parameters

### Depth Anything V2
- **Encoder**: `vitl` (Large model, best quality)
- **Colormap**: `cv2.COLORMAP_VIRIDIS` (purple→yellow) or `cv2.COLORMAP_PLASMA`

### BronchoGAN (pix2pixHD)
- **--label_nc 0**: Continuous input (depth maps, not discrete labels)
- **--no_instance**: No instance maps needed
- **--netG global**: Use global generator
- **--n_blocks_local 2**: Local enhancer blocks
- **--n_blocks_global 2**: Global generator blocks
- **--n_local_enhancers 2**: Number of local enhancers
- **--loadSize 128**: Input image resolution (128×128)

---

## Troubleshooting

### Error: "No module named 'models.depth_anything_v2'"
**Solution**: The symlink is missing. Create it:
```bash
cd /home/ahmad/BronchoGAN/pix2pixHD/models
ln -s ../../Depth-Anything-V2/depth_anything_v2 depth_anything_v2
```

### Error: "cv::ColorMap only supports CV_8UC1 or CV_8UC3"
**Solution**: Convert depth to uint8 before applying colormap:
```python
depth_vis = depth_vis.astype(np.uint8)  # Add this line!
depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
```

### Error: "test_A is not a valid directory"
**Solution**: Create proper folder structure:
```bash
mkdir -p pix2pixHD/test_data/test_A
cp your_depth.png pix2pixHD/test_data/test_A/
```

### Depth image is grayscale
**Solution**: Apply colormap for visualization:
```python
depth_colored = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
```

---

## File Structure

```
BronchoGAN/
├── Depth-Anything-V2/
│   ├── depth_anything_v2/          # Depth model code
│   ├── depth_anything_v2_vitl.pth  # Model weights (335MB) [DOWNLOAD]
│   └── test.ipynb                  # Depth extraction notebook
├── pix2pixHD/
│   ├── models/
│   │   ├── depth_anything_v2 → symlink  # Symlink to Depth-Anything-V2
│   │   └── pix2pixHD_model.py
│   ├── test_data/
│   │   └── test_A/                 # Put depth images here
│   ├── test.py                     # Test script
│   └── train_model.ipynb           # Training reference
├── Models/
│   └── latest_net_G.pth            # BronchoGAN weights (201MB) ✓
├── example.png                     # Input bronchoscopy image
├── example_depth.png               # Generated depth map
├── generated_broncho.png           # Final output
└── USAGE_GUIDE.md                  # This file
```

---

## Quick Start (TL;DR)

```bash
# 1. Convert image to depth
cd /home/ahmad/BronchoGAN/Depth-Anything-V2
# Run: See test.ipynb

# 2. Generate bronchoscopy from depth
cd /home/ahmad/BronchoGAN/pix2pixHD
mkdir -p test_data/test_A
cp ../example_depth.png test_data/test_A/

python test.py \
  --label_nc 0 --no_instance --model 'pix2pixHD' --name latest \
  --checkpoints_dir '../Models' --netG 'global' --dataroot './test_data/' \
  --n_blocks_local 2 --n_blocks_global 2 --n_local_enhancers 2 \
  --loadSize 128 --how_many 1 --results_dir './results'

# 3. Check output
open ./results/latest/test_latest/images/
```

