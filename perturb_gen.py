"""
CUDA_VISIBLE_DEVICES=0 python perturb_gen.py --arch=ResNet20-4 --data=cifar100 --epochs=50 --num_samples 10
"""

import os, sys
sys.path.insert(0, './')            # 在 './' 路径下搜索导入模块，并设为最高优先级

import argparse
import copy
import inversefed                   # 项目文件夹
import numpy as np
import random
import torch
import torch.nn.functional as F
import time

from benchmark.comm import create_model, build_transform, preprocess, create_config, vit_preprocess         # 项目文件夹
from functools import partial
from transformers import ViTFeatureExtractor, ViTForImageClassification
from inversefed.data.loss import Classification

# 没用的模块
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

from autoaugment import SubPolicy
from collections import defaultdict
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
from PIL import Image


seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

# 解析命令行
parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')         
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')          # ResNet20-4
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')        # cifar100
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')        # 50
parser.add_argument('--num_samples', default=5, type=int, help='Images per class')                  # 按类别生成对抗扰动时，每个类别中采样的数目, cifar-100中最大为100
parser.add_argument('--tiny_data', default=False, action='store_true', help='Use 0.1 training dataset')
opt = parser.parse_args()

# init training
arch = opt.arch
num_images = 1          # batch里仅有1张图片，模拟最容易被攻击的场景
trained_model = True    # 似乎没有用到这个参数

# init env
setup = inversefed.utils.system_startup()               # 硬件环境  {'device': device(type='cuda', index=0), 'dtype': torch.float32}
defs = inversefed.training_strategy('conservative')     # 训练参数  lr=0.1 epoch=None batch_size=128 optimizer='SGD' augmentations = True ...
defs.epochs = opt.epochs                                # defs.epochs=50
epsilon = 0.01
alpha = 0.7 
CIFAR100_NUM_PER_CLASS = 100


def initial_perturb():
    pass


def calculate_dw(model, imgs, labels, loss_fn, retain=False):
    model.eval()
    model.zero_grad()
    model_loss = loss_fn(model(imgs), labels)
    if retain:
        dw = torch.autograd.grad(model_loss, model.parameters(), retain_graph=True, create_graph=True)
    else:
        dw = torch.autograd.grad(model_loss, model.parameters())
    return dw


def cal_dis(a, b, metric='L2'):
    a, b = a.flatten(), b.flatten()
    if metric == 'L2':
        return torch.mean((a - b) * (a - b))
    elif metric == 'L1':
        return torch.mean(torch.abs(a-b))
    elif metric == 'cos':
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
    else:
        raise NotImplementedError


def auc_gradsim(model, loss_fn, imgs, labels, noise, ref_dw, bin_num, metric='cos', cal_grad=False):
    model.eval()
    gradsim_list = []       # 所有混合图像的GradSim值
    for dis_iter in range(bin_num+1):
        pseudo_imgs = 1.0 / bin_num * dis_iter * imgs + 1. / bin_num * (bin_num - dis_iter) * noise     # 混合图像
        if not cal_grad:
            pseudo_imgs = pseudo_imgs.detach()
        pseudo_dw = calculate_dw(model, pseudo_imgs, labels, loss_fn, retain=cal_grad)   # 混合图像的梯度
        gradsim = sum([cal_dis(dw_pse, dw_ref, metric=metric) for dw_pse, dw_ref in zip(pseudo_dw, ref_dw)]) / len(ref_dw)
        gradsim_list.append(gradsim)
    
    auc_gradsim = 0
    for i in range(bin_num):
        low = gradsim_list[i]
        high = gradsim_list[i+1]
        auc_gradsim += (low + high) / (2 * bin_num)
    return auc_gradsim


def fgsm_attack(image, epsilon, data_grad):
    """
    :param image: 需要攻击的图像
    :param epsilon: 扰动值的范围
    :param data_grad: 图像的梯度
    :return: 扰动后的图像
    """
    sign_data_grad = data_grad.sign()                       # 数据梯度的元素符号
    perturbed_image = image - epsilon*sign_data_grad        # 扰动后图像
    perturbed_image = torch.clamp(perturbed_image, 0, 1)    # 裁剪每个像素至[0,1]范围
    return perturbed_image


def denorm(batch):
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]        # 啊? why using cifar10?
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    else:
        raise NotImplementedError
    
    batch = batch.to(**setup)
    return batch*ds + dm


def stat(li):
    li_arr = np.array(li)
    li_mean = np.mean(li_arr)
    li_var = np.var(li_arr)
    score = alpha * li_mean + (1 - alpha) * li_var
    return li_mean, li_var, score


def privacy_score(idx, model, loss_fn, dataloader):
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]        # 啊? why using cifar10?
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    else:
        raise NotImplementedError
    
    init_imgs, labels = [], []              # batch，尽管每个batch中仅有1个
    init_img, label = dataloader.dataset[idx]
    init_imgs.append(init_img.to(**setup))
    labels.append(torch.as_tensor((label,), device=setup['device']))
    init_imgs = torch.stack(init_imgs)
    init_imgs.requires_grad = True
    labels = torch.cat(labels)
    
    model.eval()
    init_ref_dw = calculate_dw(model, init_imgs, labels, loss_fn, retain=True)          # 参照梯度
    metric = 'cos'
    bin_num = 20
    noise_input = (torch.rand((init_imgs.shape)).cuda() - dm) / ds          # 随机噪声图像
    
    # 图像idx的AUC(GradSim)评估值
    init_auc_gradsim = auc_gradsim(model, loss_fn, init_imgs, labels, noise_input, init_ref_dw, bin_num, metric, cal_grad=True)
    
    # fgsm attack
    model.zero_grad()
    imgs_grad = torch.autograd.grad(init_auc_gradsim, init_imgs)[0]         # 计算auc_gradsim关于输入图像的梯度
    # init_imgs_denorm = denorm(init_imgs)                                    # 对输入图像 去归一化
    # pert_imgs = fgsm_attack(init_imgs_denorm, epsilon, imgs_grad)    
    # pert_data = (pert_imgs[0], labels)      # batch size 只有1
    pert_imgs = fgsm_attack(init_imgs, epsilon, imgs_grad)
    pert_data = (pert_imgs[0], labels)          # batch size 只有1
    
    # 图像idx扰动后的AUC(GradSim)评估值
    # pert_imgs_norm = (pert_imgs - dm) / ds
    # pert_ref_dw = calculate_dw(model, pert_imgs_norm, labels, loss_fn, retain=False)            # 参照梯度
    # pert_auc_gradsim = auc_gradsim(model, loss_fn, pert_imgs_norm, labels, noise_input, pert_ref_dw, bin_num, metric, cal_grad=False)
    pert_ref_dw = calculate_dw(model, pert_imgs, labels, loss_fn, retain=False)            # 参照梯度
    pert_auc_gradsim = auc_gradsim(model, loss_fn, pert_imgs, labels, noise_input, pert_ref_dw, bin_num, metric, cal_grad=False)
    
    return init_auc_gradsim, pert_auc_gradsim, pert_data


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


def accuracy_metric(idx_list, model, dataset):
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]        # 啊? why using cifar10?
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    else:
        raise NotImplementedError
    
    imgs, labels = [], []
    
    for idx in idx_list:
        img, label = dataset[idx]
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            imgs.append(img.to(**setup))
    
    imgs = torch.stack(imgs)
    labels = torch.cat(labels)
    
    model.zero_grad()
    jacobs, labels= get_batch_jacobian(model, imgs, labels)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    
    return eval_score(jacobs, labels)


if __name__ == '__main__':
    # 加载数据：validloader经过 ToTensor(); Normalize() 处理
    trainset, validset = _build_cifar100('~/data/')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=defs.batch_size,
                    shuffle=True, drop_last=False, num_workers=4, pin_memory=True)              # trainloader就没用到过，可删
    validloader = torch.utils.data.DataLoader(validset, batch_size=defs.batch_size,
                    shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    
    # 加载模型
    loss_fn = Classification()
    model = create_model(opt)           # randomly initialized model for accuracy quantification
    model.to(**setup)                   # 将模型加载到设备，等价于 model.to(device=device(type='cuda', index=0), dtype=torch.float32)
    old_state_dict = copy.deepcopy(model.state_dict())
    model.load_state_dict(torch.load('checkpoints/tiny_data_{}_arch_{}/{}.pth'.format(opt.data, opt.arch, opt.epochs)))     # 加载评估模型
    model.eval()
    
    # 在validloader中，对所有样本按照类别进行整理，cifar-100中validset的 label的范围是[0, ..., 99]; 每个类别的样本数目为100
    sample_list = {}
    mapping = {}
    if opt.data == 'cifar100':
        num_classes = 100
    else:
        print('dataset invalid!')
        exit(1)
    for i in range(num_classes):
        sample_list[i] = []         # {0: [], ..., 99: []}
    for idx, (_, label) in enumerate(validloader.dataset):
        if isinstance(label, torch.Tensor):
            label = label.item()
        sample_list[label].append(idx)
        mapping[idx] = label * CIFAR100_NUM_PER_CLASS + len(sample_list[label]) - 1
        
    # 计算每一类图像的 adversarial perturbation
    # num_samples = opt.num_samples
    # for label in range(num_classes):
    #     # idx = sample_list[label][0 ~ num_samples-1]
    #     # img, label = validloader.dataset[idx]
        
    #     perturb_tensor = initial_perturb()  # 初始化 adversarial perturbation
        
    #     print(perturb_tensor)
    
    # 按照FGSM方法对每一张图片生成一个对抗扰动；
    # 测评时，对每一张施加对抗扰动的图像均测量privacy score；将施加扰动后的图像存储，然后仿照ATSP采样进行Accuracy Score的衡量。
    init_auc_gradsims = []      # 未施加扰动时，数据集中每个图像对应的AUC(GradSim)
    pert_auc_gradsims = []      # 施加扰动后，数据集中每个图像对应的AUC(GradSim)
    perturbed_dataset = []      # (img, label) 施加扰动后的数据集
    
    print('----------------Privacy Evaluation Start! FGSM epsilon: {}; Alpha: {}----------------'.format(epsilon, alpha))
    pri_eval_start = time.time()
    
    # 均值、方差、alpha belta 权重，score的归一化，acc score的均值；图像的归一化; time cost
    for label in range(num_classes):
    # for label in range(3):
        # num_samples = len(sample_list[label])
        class_init_auc_gradsims = []
        class_pert_auc_gradsims = []
        class_perturbed_dataset = []
        start_time = time.time()
        for idx in sample_list[label]:
            init_auc_gradsim, pert_auc_gradsim, pert_data = privacy_score(idx, model, loss_fn, validloader)
            class_init_auc_gradsims.append(init_auc_gradsim.item())
            class_pert_auc_gradsims.append(pert_auc_gradsim.item())
            class_perturbed_dataset.append(pert_data)
        # 按类别统计
        init_class_mean, init_class_var, init_class_pri_score = stat(class_init_auc_gradsims)
        pert_class_mean, pert_class_var, pert_class_pri_score = stat(class_pert_auc_gradsims)
        time_cost = time.time() - start_time
        # 可视化 包括输出进度，类别统计，样本对比
        print("-> class {}\tCompleted! Time Cost: {} s".format(label, int(time_cost)))
        print("\t- initial:\tclass_mean: {}\tclass var: {}\tclass privacy score: {}".format('%.4f' % init_class_mean, '%.4f' % init_class_var, '%.4f' % init_class_pri_score))
        print("\t- perturb:\tclass_mean: {}\tclass var: {}\tclass privacy score: {}".format('%.4f' % pert_class_mean, '%.4f' % pert_class_var, '%.4f' % pert_class_pri_score))
        
        save_path = 'show_pics/epsilon-{}/'.format(epsilon)
        init_im, _ = validloader.dataset[sample_list[label][0]]
        init_im_denorm = denorm(init_im)
        pert_im_denorm = denorm(class_perturbed_dataset[0][0])
        torchvision.utils.save_image(init_im_denorm, save_path+'{}-init.png'.format(label))
        torchvision.utils.save_image(pert_im_denorm, save_path+'{}-pert.png'.format(label))
        
        init_auc_gradsims.extend(class_init_auc_gradsims)
        pert_auc_gradsims.extend(class_pert_auc_gradsims)
        perturbed_dataset.extend(class_perturbed_dataset)
    
    init_auc_gradsims_mean, init_auc_gradsims_var, init_pri_score = stat(init_auc_gradsims)
    pert_auc_gradsims_mean, pert_auc_gradsims_var, pert_pri_score = stat(pert_auc_gradsims)
    print("<== Summary ==>")
    print("\t- initial privacy score: {}\tmean: {}\tvar: {}".format('%.4f' % init_pri_score, '%.4f' % init_auc_gradsims_mean, '%.4f' % init_auc_gradsims_var))
    print("\t- perturb privacy score: {}\tmean: {}\tvar: {}".format('%.4f' % pert_pri_score, '%.4f' % pert_auc_gradsims_mean, '%.4f' % pert_auc_gradsims_var))
    pri_eval_time_cost = time.time() - pri_eval_start
    print('----------------Privacy Evaluation Completed! Time Cost: {} s----------------'.format(int(pri_eval_time_cost)))
    
    print('#\n#')
    print("Accuracy Score Evaluation Start!")
    acc_eval_start = time.time()
    # accuracy score evaluation
    # 均值，可视化， 时间, 扰动后的图 totensor()，归一化, 可选是否评价 acc_scores_init
    model.load_state_dict(old_state_dict)
    acc_scores_init = []
    acc_scores_pert = []
    test_same = True
    for run in range(10):
        run_start = time.time()
        init_samples = [200 + run * 100 + i for i in range(100)]
        pert_samples = [mapping[200 + run * 100 + i] for i in range(100)]
        
        # test 是否是相同的图
        if test_same:
            torchvision.utils.save_image(validloader.dataset[init_samples[50]][0], 'test-same-init.png')
            torchvision.utils.save_image(perturbed_dataset[pert_samples[50]][0], 'test-same-pert.png')
            test_same = False
        
        acc_score_init = accuracy_metric(init_samples, model, validloader.dataset)
        acc_score_pert = accuracy_metric(pert_samples, model, perturbed_dataset)
        acc_scores_init.append(acc_score_init)
        acc_scores_pert.append(acc_score_pert)
        
        run_time = time.time() - run_start
        print('run {}; time cost {} s; pace {}%'.format(run, int(run_time), 10*run + 10))
    
    acc_score_init = np.mean(np.array(acc_scores_init))
    acc_score_pert = np.mean(np.array(acc_scores_pert))
    
    acc_eval_time = time.time() - acc_eval_start
    print("Accuracy Score Evaluation Completed! Time cost: {} s".format(int(acc_eval_time)))
    print("Initial Accuracy Score: ", '%.4f' % acc_score_init)
    print("Perturb Accuracy Score: ", '%.4f' % acc_score_pert)
        
    # accuracy score 采样5张取近似；原数据的acc，改后数据的acc
    # 对比原模型搜出来最好的
    
