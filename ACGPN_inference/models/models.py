import torch

def create_model(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            print('pix2pixHD')
            model = Pix2PixHDModel()
            model.to(device)
        else:
            print('InferenceModel')
            model = InferenceModel()

    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        print(f'opt.gpu_ids: {opt.gpu_ids}')
        # model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
