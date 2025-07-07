import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.transforms.functional as TF

import sys
import os
import cv2
import numpy as np
from skimage import filters, morphology

# Add the segment_script directory to the Python path
segment_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'segment_script')
if segment_script_path not in sys.path:
    sys.path.append(segment_script_path)

from lr_aspp import LiteRASPP
from .util import ZSegmentationExtractor
from .depth_anything_v2.dpt import DepthAnythingV2


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_dice_loss=False): # to use the dice or not
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, use_dice_loss)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake): # dice_loss):
            # return [l for (l,f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake, dice_loss), flags) if f]
            return [l for (l,f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake), flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.segmentation_model_path = './segment_model/fine_tuning.pt'
        self.segmentation_model = LiteRASPP.load(self.segmentation_model_path, self.device).eval()
        # Initialize DepthAnything model
        self.depth_model = self._setup_depth_model()
        self.segmentation_extractor = ZSegmentationExtractor(
            spatial_dim=128,  # adjust based on your needs
            watershed_compactness=3,
            avg_pool_kernel_size=1,
            corner_margin=25,
            edge_margin=7,
            intensity_threshold=0.5
        ).to(device=self.device)

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                  
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
        
            # Names so we can breakout loss
            # self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake', 'Dice_loss')
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)         
        input_label = input_label

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)  

        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)                     
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label
            
        fake_image = self.netG.forward(input_concat)
        
        
        #### Segemtation ######
        
        # Segmentation of the input images
        binary_mask = self.segment_image(real_image)
        
        # Segmentation of the generated fake_image
        segmented_mask = self.segment_image(fake_image)

        # Calculate dice loss here using binary_mask and segmented_mask
        dice_loss = self.dice_loss(binary_mask, segmented_mask)

        with torch.no_grad():
            # Process input label directly (skip depth estimation)
            input_binary_mask = self.process_depth_and_segment(input_label, is_input_label=True)
            
            # Process fake/generated image through depth estimation
            fake_binary_mask = self.process_depth_and_segment(fake_image, is_input_label=False)
            
            # Calculate dice loss
            dice_loss = self.dice_loss(input_binary_mask, fake_binary_mask)
            
        #########################

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        
        # # Only return the fake_B image if necessary to save BW
        # return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, dice_loss),  
        #         fake_image if infer else None,
        #         input_binary_mask if infer else None,
        #         fake_binary_mask if infer else None
        #         ]

        # Only return the fake_B image if necessary to save BW
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake),  
                fake_image if infer else None,
                ]
        
    def segment_image(self, image):
        # Normalize the RGB image (if normalization is necessary)
        mu, sigma = self.calculate_mean_std(image)
        normalized_image = (image - mu.view(1, 3, 1, 1)) / sigma.view(1, 3, 1, 1)

        # Process with segmentation model
        with torch.no_grad():
            logits = self.segmentation_model(normalized_image)
            softmax_scores = F.interpolate(logits, size=image.size()[2:], mode='bilinear').squeeze().softmax(0)[1]
            mask = (softmax_scores >= 0.55).float().unsqueeze(0)  # Add a channel dimension
            mask = mask.repeat(1, 3, 1, 1)  # Repeat to match the RGB channels
            # print("mask_function", mask.size())
        return mask

    def calculate_mean_std(self, images):
        channels_flat = images.view(images.size(0), images.size(1), -1)
        mean = channels_flat.mean(2).mean(0)
        std = channels_flat.std(2).mean(0)
        return mean, std
            

    def inference(self, label, inst, image=None):
        # Encode Inputs        
        image = image if image is not None else None
        with torch.no_grad():
            input_label, inst_map, real_image, _ = self.encode_input(label, Variable(inst), image, infer=True)

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features             
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label        
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        
        # Segmentation of the generated fake_image
        segmented_mask = self.segment_image(fake_image)

        return fake_image, segmented_mask
        # return fake_image 

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.AdamW(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
    
    def dice_loss(self, true_mask, pred_mask, smooth=1.0):
        # Ensure inputs are in the right shape
        true_mask = true_mask.view(true_mask.size(0), -1)  # [B, N]
        pred_mask = pred_mask.view(pred_mask.size(0), -1)  # [B, N]
        
        # Calculate intersection and union
        intersection = (true_mask * pred_mask).sum(dim=1)
        union = true_mask.sum(dim=1) + pred_mask.sum(dim=1)
        
        # Calculate Dice coefficient for each item in batch
        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Return mean loss
        return 1 - dice.mean()
    

    #### Segment images using depth any thing ##### 
    def _setup_depth_model(self):
            model_configs = {
                'vitl': {
                    'encoder': 'vitl',
                    'features': 256,
                    'out_channels': [256, 512, 1024, 1024]
                }
            }
            model = DepthAnythingV2(**model_configs['vitl'])
            current_dir = os.path.dirname(os.path.abspath(__file__))

            weights_path = os.path.join(current_dir, 'depth_anything_v2_vitl.pth')
            model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            return model.to(self.device).eval()

    def get_filtered(self, image, cutoffs, squared_butterworth=False, order=5.0, npad=211):
        highpass_filtered = []
        for cutoff in cutoffs:
            highpass_filtered.append(
                filters.butterworth(
                    image,
                    cutoff_frequency_ratio=cutoff,
                    order=order,
                    high_pass=True,
                    squared_butterworth=squared_butterworth,
                    npad=npad,
                )
            )
        return highpass_filtered

    def process_depth_and_segment(self, image, is_input_label=False):
        """
        Process image through depth estimation and segmentation
        Args:
            image: Input tensor (either generated image or input label)
            is_input_label: Boolean flag to indicate if the input is a label image
        """
        with torch.no_grad():
            if isinstance(image, torch.Tensor):
                if image.dim() == 3:
                    image = image.unsqueeze(0)

                if is_input_label:
                    # For input label, directly use it as depth map
                    depth_tensor = image[0][0].to(self.device)  # Take first channel as depth
                else:
                    # For generated images, run through depth estimation
                    image_np = image.cpu().numpy()
                    image_np = np.transpose(image_np[0], (1, 2, 0))
                    resized_img = cv2.resize(image_np, (128, 128))
                    
                    # Get depth prediction
                    depth = self.depth_model.infer_image(resized_img)
                    max_depth = depth.max()
                    inverted_depth = max_depth - depth
                    
                    # Apply high-pass filter
                    cutoffs = [0.0126, 0.05, 0.32]
                    highpass_depths = self.get_filtered(inverted_depth, cutoffs)
                    depth_tensor = torch.from_numpy(highpass_depths[0]).to(self.device)

                # Get segmentation using the depth map
                segmentation_data = self.segmentation_extractor.extract_segmentation(
                    depth_tensor, 
                    rgb_img=None,  # No need for RGB in this case
                    return_plot_data=True
                )
                
                # Get binary mask and clean it
                seg_mask = segmentation_data['seg_mask']
                binary_mask = (seg_mask > 0).float()
                
                # Handle dimensions
                if binary_mask.dim() == 2:
                    binary_mask = binary_mask.unsqueeze(0)
                elif binary_mask.dim() == 3 and binary_mask.size(0) != 3:
                    binary_mask = binary_mask.repeat(3, 1, 1)
                
                # Resize if needed
                if binary_mask.shape[-2:] != image.shape[-2:]:
                    binary_mask = F.interpolate(
                        binary_mask.unsqueeze(0),
                        size=(image.shape[2], image.shape[3]),
                        mode='nearest'
                    ).squeeze(0)
                
                return binary_mask.to(self.device)

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
