from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    """
    Converts a tensor into a numpy array image.
    
    Args:
        image_tensor (torch.Tensor): The input tensor
        imtype (numpy dtype): The desired output numpy dtype
        normalize (bool): Whether to normalize the tensor
    Returns:
        numpy.ndarray: The converted image
    """
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    # Handle the case where input is already numpy
    if isinstance(image_tensor, np.ndarray):
        return image_tensor.astype(imtype)

    # Get tensor onto CPU
    image_tensor = image_tensor.cpu().float()

    # Handle different tensor shapes
    if image_tensor.dim() == 2:  # Single channel, HxW
        image_tensor = image_tensor.unsqueeze(0)  # Add channel dim: 1xHxW

    # If input is a binary mask (1xHxW)
    if image_tensor.size(0) == 1:
        # Check if it's a binary mask
        if image_tensor.max() <= 1 and not normalize:
            image_numpy = image_tensor.numpy()
            # Reorder dimensions to HxWx1
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            # Scale to 0-255 range
            image_numpy = image_numpy * 255.0
            # Remove single channel dimension if present
            if image_numpy.shape[-1] == 1:
                image_numpy = image_numpy.squeeze(-1)
            return image_numpy.astype(imtype)

    # Regular image processing (3xHxW)
    image_numpy = image_tensor.numpy()
    
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
        
    image_numpy = np.clip(image_numpy, 0, 255)
    
    # Handle channel dimension
    if image_numpy.shape[2] == 1:  # Single channel
        image_numpy = image_numpy.squeeze(2)
    elif image_numpy.shape[2] > 3:  # More than 3 channels
        image_numpy = image_numpy[:, :, 0]
        
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# convert instance segmetation to binary
def to_binary_mask(tensor):
    # Ensure tensor is on CPU
    tensor = tensor.cpu()

    # Handle channels (assuming the tensor is in BxCxHxW format and channels are last after squeeze)
    if tensor.size(1) == 4:  # RGB + alpha
        tensor = tensor[:, :3, :, :]  # Take only RGB channels

    # Create a binary mask where any channel has a value greater than 0
    binary_mask = torch.any(tensor > 0, dim=1, keepdim=True).type(torch.uint8)
    

    return binary_mask
###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
