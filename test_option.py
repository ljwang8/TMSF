from argparse import ArgumentParser

import datasets.CD_dataset
from evaluator import *

print(torch.cuda.is_available())

"""
eval the CD model
"""

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--calculate_flops', default=False, help="Enable multimodal input (jsonA and jsonB)")
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--print_models', default=False, type=bool, help='print models')
    parser.add_argument('--is_TMSF', default=True)


    #选择数据集
    parser.add_argument('--data_name', default='LEVIR', type=str)
    # parser.add_argument('--data_name', default='CDD', type=str)
    # parser.add_argument('--data_name', default='WHUCD', type=str)

    #设置存放模型文件的目录
    parser.add_argument('--checkpoint_root', default='./checkpoints/LEVIR', type=str)
    parser.add_argument('--project_name', default='MT_res18_stages4_prefus_damtoken_trans_fuse1_1119', type=str)
    #选择模型名
    parser.add_argument('--net_G', default='MT_res18_stages4_prefus_damtoken_trans_fuse1', type=str,
                        help=
                        # ours
                        'MT_res18_stages3_prefus_damtoken_trans_fuse1 | '
                        # Ablation on DAT-based Transformer
                        'MT_res18_stages4_prefus_damtoken_trans_fuse1 | MT_res18_stages4_prefus_trans_fuse1| MT_res18_stages4_prefus_damtoken_fuse1'
                        # Ablation on Multiscale Feature Adaptive Fusion Module
                        'MT_res18_stages4_prefus_damtoken_trans_fuse0 | MT_res18_stages4_prefus_damtoken_trans_fuse2 | MT_res18_stages4_prefus_damtoken_trans_fuse3 |MT_res18_stages4_prefus_damtoken_trans_fuse4'
                        # comparison methods
                        'FC_EF | SiamUnet_diff | SiamUnet_conc| '
                        'A2Net | SNUNet_ECAM | DTCDSCN '
                        'BIT_stages4_ed1_dd8_ddm8 | AMT | ChangeFormerV6 | RCTNet'
                        )
    #可视化路径
    parser.add_argument('--vis_root', default='vis_LEVIR', type=str)
    # parser.add_argument('--vis_root', default='vis_CDD', type=str)
    # parser.add_argument('--vis_root', default='vis_WHUCD', type=str)

    #一般超参数
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--n_class', default=2, type=int)


    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)
    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    dataloader = datasets.CD_dataset.get_loader(args.data_name, img_size=args.img_size,
                                                batch_size=args.batch_size, is_train=False,
                                                split=args.split)
    model = CDEvaluator(args=args, dataloader=dataloader)

    #测试集
    model.eval_models(checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    main()

