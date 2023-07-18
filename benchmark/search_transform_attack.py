"""
CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=13-43-18 --mode=aug --arch=ResNet20-4 --data=cifar100 --epochs=50
"""

import os, sys
sys.path.insert(0, './')            # 在 './' 路径下搜索导入模块，并设为最高优先级

import argparse
import copy
import inversefed                   # 项目文件夹
import numpy as np
import policy                       # 项目文件
import random
import torch
import torch.nn.functional as F

from benchmark.comm import create_model, build_transform, preprocess, create_config, vit_preprocess         # 项目文件夹
from functools import partial
from transformers import ViTFeatureExtractor, ViTForImageClassification

# 没用的模块
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from autoaugment import SubPolicy
from collections import defaultdict
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
from PIL import Image


policies = policy.policies          # 候选子策略库，共50个
seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

# 解析命令行
parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')         
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')                  # aug
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')      # xx-xx-xx
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')          # ResNet20-4
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')        # cifar100
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')        # 50
parser.add_argument('--num_samples', default=5, type=int, help='Images per class')
parser.add_argument('--tiny_data', default=False, action='store_true', help='Use 0.1 training dataset')
opt = parser.parse_args()

# init training
arch = opt.arch
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']
num_images = 1          # batch里仅有1张图片，模拟最容易被攻击的场景
trained_model = True    # 似乎没有用到这个参数

# init env
setup = inversefed.utils.system_startup()               # 硬件环境  {'device': device(type='cuda', index=0), 'dtype': torch.float32}
defs = inversefed.training_strategy('conservative')     # 训练参数  lr=0.1 epoch=None batch_size=128 optimizer='SGD' augmentations = True ...
defs.epochs = opt.epochs                                # defs.epochs=50


def similarity_measures(img_batch, ref_batch, batched=True, method='fsim'):
    """提取一批图像之间的 批相似度"""
    from image_similarity_measures.quality_metrics import fsim, issm, rmse, sam, sre, ssim, uiq
    methods = {'fsim':fsim, 'issm':issm, 'rmse':rmse, 'sam':sam, 'sre':sre, 'ssim':ssim, 'uiq':uiq }

    def get_similarity(img_in, img_ref):                # 单张图片的相似度
        return methods[method](img_in.permute(1,2,0).numpy(), img_ref.permute(1,2,0).numpy())
        
    if not batched:
        sim = get_similarity(img_batch.detach(), ref_batch)     # .detach() 起别名；同时梯度截断
    else:
        [B, C, m, n] = img_batch.shape
        sim_list = []
        for sample in range(B):
            sim_list.append(get_similarity(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))

        sim_list = np.array(sim_list)
        sim_list = sim_list[~np.isnan(sim_list)]        # 将sim_list中为空值(np.nan)的剔除
        sim = np.mean(sim_list)
    return sim


def collate_fn(examples, label_key='fine_label'):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[label_key] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))


def get_batch_jacobian(net, x, target):
    net.eval()
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    if isinstance(y, tuple):
        y = y[0] # vit model return  (logit)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach()


def calculate_dw(model, inputs, labels, loss_fn):
    model.zero_grad()
    target_loss = loss_fn(model(inputs), labels)
    dw = torch.autograd.grad(target_loss, model.parameters())
    return dw


def cal_dis(a, b, metric='L2'):
    a, b = a.flatten(), b.flatten()
    if metric == 'L2':
        return torch.mean((a - b) * (a - b)).item()
    elif metric == 'L1':
        return torch.mean(torch.abs(a-b)).item()
    elif metric == 'cos':
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    else:
        raise NotImplementedError


def accuracy_metric(idx_list, model, loss_fn, trainloader, validloader, label_key='fine_label'):
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    elif opt.data == 'ImageNet':
        dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
    elif opt.data.startswith('CelebA'):
        dm = torch.as_tensor(inversefed.consts.celeba_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.celeba_std, **setup)[:, None, None]
    else:
        raise NotImplementedError

    # prepare data
    ground_truth, labels = [], []

    if isinstance(model, ViTForImageClassification):
        #return tuple(logits,) instead of ModelOutput object
        model.forward = partial(model.forward, return_dict=False)
        for idx in idx_list:
            example = validloader.dataset[idx]
            label = example[label_key]

            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(example)
        
        ground_truth = collate_fn(ground_truth, label_key=label_key)['pixel_values'].to(**setup)

    else: 
        for idx in idx_list:    # 在idx_list中出现的所有类别中，每一类取一对(img, label)
            img, label = validloader.dataset[idx]
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(img.to(**setup))

        ground_truth = torch.stack(ground_truth)
        
    labels = torch.cat(labels)
    model.zero_grad()
    jacobs, labels= get_batch_jacobian(model, ground_truth, labels)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    return eval_score(jacobs, labels)


def reconstruct(idx, model, loss_fn, trainloader, validloader, label_key='fine_label'):
    # dm, ds用于随机噪声图像的标准化
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]        # 啊? why using cifar10?
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    elif opt.data == 'ImageNet':
        dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
    elif opt.data.startswith('CelebA'):
        dm = torch.as_tensor(inversefed.consts.celeba_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.celeba_std, **setup)[:, None, None]
    else:
        raise NotImplementedError
    
    # prepare data
    ground_truth, labels = [], []
    if isinstance(model, ViTForImageClassification):
        #return tuple(logits,) instead of ModelOutput object
        model.forward = partial(model.forward, return_dict=False)
        while len(labels) < num_images:
            example = validloader.dataset[idx]
            label = example[label_key]

            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(example)
        
        ground_truth = collate_fn(ground_truth, label_key=label_key)['pixel_values'].to(**setup)

    else: 
        while len(labels) < num_images:             # num_images=1 batch里仅有1张图像
            img, label = validloader.dataset[idx]
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(img.to(**setup))

        ground_truth = torch.stack(ground_truth)

    labels = torch.cat(labels)
    model.zero_grad()
    # calcuate ori dW
    target_loss = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())

    metric = 'cos'

    # attack model
    model.eval()
    dw_list = list()
    dx_list = list()
    bin_num = 20
    noise_input = (torch.rand((ground_truth.shape)).cuda() - dm) / ds       # 随机噪声图像
    for dis_iter in range(bin_num+1):       # [0, 1, ..., 20]
        model.zero_grad()
        fake_ground_truth = (1.0 / bin_num * dis_iter * ground_truth + 1. / bin_num * (bin_num - dis_iter) * noise_input).detach()      # 混合图像
        fake_dw = calculate_dw(model, fake_ground_truth, labels, loss_fn)
        dw_loss = sum([cal_dis(dw_a, dw_b, metric=metric) for dw_a, dw_b in zip(fake_dw, input_gradient)]) / len(input_gradient)

        dw_list.append(dw_loss)

    interval_distance = cal_dis(noise_input, ground_truth, metric='L1') / bin_num


    def area_ratio(y_list, inter):
        # 这 inter 参数似乎没起作用啊？interval_distance感觉完全没必要计算
        area = 0
        max_area = inter * bin_num
        for idx in range(1, len(y_list)):
            prev = y_list[idx-1]
            cur = y_list[idx]
            area += (prev + cur) * inter / 2
        return area / max_area

    return area_ratio(dw_list, interval_distance)



def main():
    if opt.arch not in ['vit']:         # opt.arch=ResNet20-4   opt.data=cifar100
        loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)       # trainloader 数据未经过转换    validloader 为转换后的数据
        model = create_model(opt)       # randomly initialized model for accuracy quantification
    else:
        loss_fn, trainloader, validloader, model, mean_std, scale_size = vit_preprocess(opt, defs, valid=True)      # batch size rescale to 16
    model.to(**setup)                   # 等价于 model.to(device=device(type='cuda', index=0), dtype=torch.float32), 将模型加载到设备上
    old_state_dict = copy.deepcopy(model.state_dict())                                                                      # 用于accuracy score
    model.load_state_dict(torch.load('checkpoints/tiny_data_{}_arch_{}/{}.pth'.format(opt.data, opt.arch, opt.epochs)))     # 用于privacy score
    model.eval()

    import time
    start = time.time()

    compute_privacy_score = True
    compute_acc_score = True

    sample_list = {}    # 在validloader中，对所有样本按照类别进行整理
    label_key = 'fine_label' if opt.data == 'cifar100' else 'label'     # 在cifar100中用不到
    if opt.data == 'cifar100':
        num_classes = 100
    elif opt.data == 'FashionMinist':
        num_classes = 10
    elif opt.data == 'ImageNet':
        num_classes = 25
    elif opt.data in ['CelebA_Gender', 'CelebA_Smile']:
        num_classes = 2
    elif opt.data == 'CelebA_Identity':
        num_classes = 100
    elif opt.data == 'CelebA_MLabel':
        num_classes = 40
    elif opt.data == 'CelebAFaceAlign_MLabel':
        num_classes = 40
    for i in range(num_classes):
        sample_list[i] = []     # {0: [], ..., 99: []}
    if opt.arch not in ['vit']:
        for idx, (_, label) in enumerate(validloader.dataset):   
            if isinstance(label, torch.Tensor):
                label = label.item()
            sample_list[label].append(idx)
    else:
        for idx, sample in enumerate(validloader.dataset):   
            sample_list[sample[label_key]].append(idx)

    if compute_privacy_score:
        metric_list = list()                # privacy score metric list
        num_samples = opt.num_samples       # images per class, default: 5
        for label in range(num_classes):    # 每个类别中取 5 张图片进行评估，将评估结果(取均值)存入 metric_list=[ , ..., ]
            metric = []
            for idx in range(num_samples):
                metric.append(reconstruct(sample_list[label][idx], model, loss_fn, trainloader, validloader, label_key))
                # print('attach {}th in class {}, auglist:{} metric {}'.format(idx, label, opt.aug_list, metric))
            metric_list.append(np.mean(metric,axis=0))

        pathname = 'search/data_{}_arch_{}/{}'.format(opt.data, opt.arch, opt.aug_list)
        root_dir = os.path.dirname(pathname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if len(metric_list) > 0:
            print('*Privacy Score Mean*: {}.'.format(np.mean(metric_list)))          # 所有类别评估值的总体均值
            np.save(pathname, metric_list)

    if compute_acc_score:
    # maybe need old_state_dict
        model.load_state_dict(old_state_dict)
        score_list = list()
        for run in range(10):
            large_samle_list = [200 + run  * 100 + i for i in range(100)]   # idx list [200, ..., 299] 每次100张图像
            score = accuracy_metric(large_samle_list, model, loss_fn, trainloader, validloader, label_key)
            score_list.append(score)
    
        pathname = 'accuracy/data_{}_arch_{}/{}'.format(opt.data, opt.arch, opt.aug_list)
        root_dir = os.path.dirname(pathname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        np.save(pathname, score_list)
        print('*Accuracy Score list*: {}.'.format(score_list))

    print('*Time Cost*: ', time.time() - start, 's')

if __name__ == '__main__':
    main()
