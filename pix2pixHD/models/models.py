import torch

def create_model(opt):
    if opt.model == 'pix2pixHD':
        # Modified import to use absolute import path
        try:
            from models.pix2pixHD_model import Pix2PixHDModel, InferenceModel
        except ImportError:
            # Fallback to local import if the above fails
            from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
            
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    else:
        try:
            from models.ui_model import UIModel
        except ImportError:
            from .ui_model import UIModel
        model = UIModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids) and not opt.fp16:
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
