import sys
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import os
import numpy as np
import torch
from torch.autograd import Variable
import cv2

NC = 14

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.append(current_directory)


def fashion_test():
    print(f'current_directory: {current_directory}')

    # 首先，调用generate_label_plain(input_label)函数，输入参数input_label是一个多通道的标签图像张量。函数将输入的多通道标签图像转换为单通道的标签图像，返回label_batch，其中label_batch是一个张量，形状为(batch_size, 1, 256, 192)，表示每个样本中的标签图像已经转换为单通道。
    def generate_label_plain(inputs):
        size = inputs.size()
        pred_batch = []
        for input in inputs:
            input = input.view(1, NC, 256, 192)
            pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
            pred_batch.append(pred)

        pred_batch = np.array(pred_batch)
        pred_batch = torch.from_numpy(pred_batch)
        label_batch = pred_batch.view(size[0], 1, 256, 192)

        return label_batch

    # 该函数通过将单通道的标签图像转换回多通道的标签图像，返回input_label
    def generate_label_color(inputs):
        label_batch = []
        for i in range(len(inputs)):
            label_batch.append(util.tensor2label(inputs[i], opt.label_nc))
        label_batch = np.array(label_batch)
        label_batch = label_batch * 2 - 1
        input_label = torch.from_numpy(label_batch)

        return input_label

    def complete_compose(img, mask, label):
        label = label.cpu().numpy()
        M_f = label > 0
        M_f = M_f.astype(np.int64)
        M_f = torch.FloatTensor(M_f).cuda()
        masked_img = img * (1 - mask)
        M_c = (1 - mask.cuda()) * M_f
        M_c = M_c + torch.zeros(img.shape).cuda()  ##broadcasting
        return masked_img, M_c, M_f

    def compose(label, mask, color_mask, edge, color, noise):
        # check=check>0
        # print(check)
        masked_label = label * (1 - mask)
        masked_edge = mask * edge
        masked_color_strokes = mask * (1 - color_mask) * color
        masked_noise = mask * noise
        return masked_label, masked_edge, masked_color_strokes, masked_noise

    def changearm(old_label):
        label = old_label
        arm1 = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.int64))
        arm2 = torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.int64))
        noise = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.int64))
        label = label * (1 - arm1) + arm1 * 4
        label = label * (1 - arm2) + arm2 * 4
        label = label * (1 - noise) + noise * 4
        return label

    os.makedirs('sample', exist_ok=True)
    opt = TrainOptions().parse()
    # config
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('# Inference images = %d' % dataset_size)

    print('Start Testing:')

    # model create
    model = create_model(opt)

    epoch_start_time = time.time()
    for idx, data in enumerate(dataset, start=0):
        print('Start Testing:1')

        # add gaussian noise channel
        # wash the label
        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        mask_clothes = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
        mask_fore = torch.FloatTensor((data['label'].cpu().numpy() > 0).astype(np.int64))
        img_fore = data['image'] * mask_fore
        img_fore_wc = img_fore * mask_fore
        all_clothes_label = changearm(data['label'])

        print('Start Testing:2')

        ############## Forward Pass ######################
        losses, fake_image, real_image, input_label, L1_loss, style_loss, clothes_mask, CE_loss, rgb, alpha = model(
            Variable(data['label'].cuda()), Variable(data['edge'].cuda()), Variable(img_fore.cuda()),
            Variable(mask_clothes.cuda())
            , Variable(data['color'].cuda()), Variable(all_clothes_label.cuda()), Variable(data['image'].cuda()),
            Variable(data['pose'].cuda()), Variable(data['image'].cuda()), Variable(mask_fore.cuda()))

        print('Start Testing:3')

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        print(f'losses: {losses}')

        print('Start Testing:4')

        ## display output images
        a = generate_label_color(generate_label_plain(input_label)).float().cuda()
        b = real_image.float().cuda()
        c = fake_image.float().cuda()
        d = torch.cat([clothes_mask, clothes_mask, clothes_mask], 1)
        #输出合并图
        # combine = torch.cat([a[0], d[0], b[0], c[0], rgb[0]], 2).squeeze()
        #输出单个修改图
        combine = c[0].squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2

        rgb = (cv_img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        name = data['name'][0]
        print(f'name {name}')
        cv2.imwrite('sample/' + name, bgr)
        print(idx)

    print('Start Testing:6')

    # end of epoch
    print('End of Time Taken: %d sec' % (time.time() - epoch_start_time))


if __name__ == '__main__':
    fashion_test()
