"""
    python -u searchalg/batch_generate.py  --arch=ResNet20-4 --data=cifar100 > batch_generate.sh
        python -u (unbuffered): 强制其标准输出也同标准错误一样不通过缓存直接打印到屏幕
    用于生成bash命令行
"""
import copy, random
import argparse

# 命令行解析
parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')      # 模型架构，字符串，参数必需
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')    # 数据集，字符串，参数必需
opt = parser.parse_args()


scheme_list = list()        # transform策略的列表，其中每个策略都由一个列表表示: [[policy1], [policy2], ... ]
num_per_gpu = 20            # 每个gpu cuda上跑的命令行数
gpu_num = 8                 # cuda_visible_devices 数目: 0-7
group_num = 10              # 组数


def write():
    """将策略列表转为命令行写入"""
    for i in range(len(scheme_list) // num_per_gpu):
        print('{')
        for idx in range(i*num_per_gpu, i*num_per_gpu + num_per_gpu):
            sch_list = [str(sch) for sch in scheme_list[idx]]
            suf = '-'.join(sch_list)
            cmd = 'CUDA_VISIBLE_DEVICES={} python benchmark/search_transform_attack.py --aug_list={} --mode=aug --arch={} --data={} --epochs=100'.format(i%gpu_num, suf, opt.arch, opt.data)
            print(cmd)
        print('}&')


def backtracing(num, scheme):   # num, scheme似乎都没用啊
    """策略总数为 50 + 50^2 + 50^3, 搜索的策略总数为 group_num * gpu_num * num_per_gpu = 1600 条，与论文中的 C_max =1500 也不一样啊 ..."""
    for _ in range(group_num * gpu_num * num_per_gpu):
        scheme = list()
        for i in range(3):                              # 每个策略包含3个子策略
            scheme.append(random.randint(-1, 50))       # -1 表示不采用任何子策略
        new_policy = copy.deepcopy(scheme)
        for i in range(len(new_policy)):                # 删除 -1 的子策略
            if -1 in new_policy:
                new_policy.remove(-1)
        scheme_list.append(new_policy)
    write()                                             # 将策略列表转为命令行写入


if __name__ == '__main__':
    backtracing(5, scheme=list())   # 你传参也没jb用啊
