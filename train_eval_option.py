from argparse import ArgumentParser

import datasets.CD_dataset
from trainer import *

print(torch.cuda.is_available())

#训练验证，  测试
"""
the main function for training the CD networks
"""


def train(args):
    dataloaders = datasets.CD_dataset.get_loaders(args)
    trainer = CDTrainer(args=args, dataloaders=dataloaders)
    trainer.train_models()


def test(args):
    from evaluator import CDEvaluator
    dataloader = datasets.CD_dataset.get_loader(args.data_name, img_size=args.img_size,
                                                batch_size=args.batch_size, is_train=False,
                                                split='test')
    eva = CDEvaluator(args=args, dataloader=dataloader)

    eva.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--calculate_flops', default=False, help="Enable multimodal input (jsonA and jsonB)")
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--is_TMSF', default=True)

    #选择数据集
    parser.add_argument('--data_name', default='LEVIR', type=str)
    # parser.add_argument('--data_name', default='LEVIR', type=str)
    # parser.add_argument('--data_name', default='WHUCD', type=str)
    # parser.add_argument('--data_name', default='CDD', type=str)

    #设置存放模型文件的目录
    parser.add_argument('--checkpoint_root', default='./checkpoints/LEVIR', type=str)
    parser.add_argument('--project_name', default='MT_res18_stages4_prefus_damtoken_trans_fuse1_0508', type=str)
    #选择模型名
    parser.add_argument('--net_G', default='MT_res18_stages4_prefus_trans_fuse1', type=str,
                        help=
                             #ours
                             'MT_res18_stages3_prefus_damtoken_trans_fuse1 | '
                             #Ablation on DAT-based Transformer
                             'MT_res18_stages4_prefus_damtoken_trans_fuse1 | MT_res18_stages4_prefus_trans_fuse1| MT_res18_stages4_prefus_damtoken_fuse1'
                             #Ablation on Multiscale Feature Adaptive Fusion Module
                             'MT_res18_stages4_prefus_damtoken_trans_fuse0 | MT_res18_stages4_prefus_damtoken_trans_fuse2 | MT_res18_stages4_prefus_damtoken_trans_fuse3 |MT_res18_stages4_prefus_damtoken_trans_fuse4'
                             #comparison methods
                             'FC_EF | SiamUnet_diff | SiamUnet_conc| '
                             'A2Net | SNUNet_ECAM | DTCDSCN '
                             'BIT_stages4_ed1_dd8_ddm8 | AMT | ChangeFormerV6 | RCTNet'
                        )
    #可视化路径
    parser.add_argument('--vis_root', default='vis_LEVIR', type=str)
    # parser.add_argument('--vis_root', default='vis_CDD', type=str)
    # parser.add_argument('--vis_root', default='vis_WHUCD', type=str)

    #一般超参数
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)
    parser.add_argument('--compare_metric', default="mf1", type=str, help='mf1 | F1_1 ')
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--multi_scale_train', default=False, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs = '+', type = float, default = [0.5, 0.5, 0.5, 0.8, 1.0])

    #损失函数
    parser.add_argument('--loss', default='ce', type=str)
    #优化器
    parser.add_argument('--optimizer', default='adamw', type=str)
    #学习率调整策略
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=100, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    #训练集和验证集
    train(args)
    #测试集
    test(args)
