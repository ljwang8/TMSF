import torch
from torch.nn import init

# from models.AMT.AMT_network import AMTNet
# from models.BIT import BASE_Transformer, ResNet
# from models.DSIFN import DSIFN
from models.TMSF import MutiscaleTransformer
# from models.SNUNet_ECAM import SNUNet_ECAM
# from models.ChangeFormer.ChangeFormer import ChangeFormerV1, ChangeFormerV2, ChangeFormerV3, ChangeFormerV4, ChangeFormerV5, ChangeFormerV6
# from models.SiamUnet_diff import SiamUnet_diff
# from models.SiamUnet_conc import SiamUnet_conc
# from models.Unet import Unet
# from models.DTCDSCN import CDNet34
# from models.A2Net.a2net import A2Net
# from models.RCTNet.rctnet import RCTNet





#权重初始化函数：定义了如何初始化网络中各种类型层的权重和偏置的方法
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):

    #ours
    if args.net_G == "MT_res18_stages4_prefus_damtoken_trans_fuse1":
        net = MutiscaleTransformer(backbone='resnet18', resnet_stages_num=4, embedding_dim=32, pre_fusion=True, token_len=8, use_dat=True, use_transfromer=True, enc_depth=2, dec_depth=8, edm=64, ddm=8, post_fusion=1,
                                   configname='MT_res18_stages4_prefus_damtoken_trans_fuse1')

    #Ablation on DAT-based Transformer
    elif args.net_G == "MT_res18_stages4_prefus_trans_fuse1":
        net = MutiscaleTransformer(backbone='resnet18', resnet_stages_num=4, embedding_dim=32, pre_fusion=True, token_len=8, use_dat=False, use_transfromer=True, enc_depth=2, dec_depth=8, edm=64, ddm=8, post_fusion=1,
                                   configname='MT_res18_stages4_prefus_trans_fuse1')
    elif args.net_G == "MT_res18_stages4_prefus_damtoken_fuse1":
        net = MutiscaleTransformer(backbone='resnet18', resnet_stages_num=4, embedding_dim=32, pre_fusion=True, token_len=8, use_dat=True, use_transfromer=False, post_fusion=1,
                                   configname='MT_res18_stages4_prefus_damtoken_fuse1')

    #Ablation on Multiscale Feature Adaptive Fusion Module
    elif args.net_G == "MT_res18_stages4_prefus_damtoken_trans_fuse0":
        net = MutiscaleTransformer(backbone='resnet18', resnet_stages_num=4, embedding_dim=32, pre_fusion=True, token_len=8, use_dat=True, use_transfromer=True, enc_depth=2, dec_depth=8, edm=64, ddm=8, post_fusion=0,
                                   configname='MT_res18_stages4_prefus_damtoken_trans_fuse0')
    elif args.net_G == "MT_res18_stages4_prefus_damtoken_trans_fuse2":
        net = MutiscaleTransformer(backbone='resnet18', resnet_stages_num=4, embedding_dim=32, pre_fusion=True, token_len=8, use_dat=True, use_transfromer=True, enc_depth=2, dec_depth=8, edm=64, ddm=8, post_fusion=2,
                                   configname='MT_res18_stages4_prefus_damtoken_trans_fuse2')
    elif args.net_G == "MT_res18_stages4_prefus_damtoken_trans_fuse3":
        net = MutiscaleTransformer(backbone='resnet18', resnet_stages_num=4, embedding_dim=32, pre_fusion=True, token_len=8, use_dat=True, use_transfromer=True, enc_depth=2, dec_depth=8, edm=64, ddm=8, post_fusion=3,
                                   configname='MT_res18_stages4_prefus_damtoken_trans_fuse3')
    elif args.net_G == "MT_res18_stages4_prefus_damtoken_trans_fuse4":
        net = MutiscaleTransformer(backbone='resnet18', resnet_stages_num=4, embedding_dim=32, pre_fusion=True, token_len=8, use_dat=True, use_transfromer=True, enc_depth=2, dec_depth=8, edm=64, ddm=8, post_fusion=4,
                                   configname='MT_res18_stages4_prefus_damtoken_trans_fuse4')



    # # comparison methods
    # elif args.net_G == "FC_EF":
    #     #Usually abbreviated as FC-EF = Image Level Concatenation
    #     #Implementation of ``Fully convolutional siamese networks for change detection''
    #     #Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
    #     net = Unet(input_nbr=3, label_nbr=2)
    # elif args.net_G == "SiamUnet_diff":
    #     #Implementation of ``Fully convolutional siamese networks for change detection''
    #     #Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
    #     net = SiamUnet_diff(input_nbr=3, label_nbr=2)
    # elif args.net_G == "SiamUnet_conc":
    #     #Implementation of ``Fully convolutional siamese networks for change detection''
    #     #Code copied from: https://github.com/rcdaudt/fully_convolutional_change_detection
    #     net = SiamUnet_conc(input_nbr=3, label_nbr=2)
    # elif args.net_G == 'base_resnet18':
    #     net = ResNet(input_nc=3, output_nc=2, backbone='resnet18',output_sigmoid=False)
    # elif args.net_G == 'BIT_stages4_ed1_dd1_ddm64':
    #     net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                            with_pos='learned')
    # elif args.net_G == 'BIT_stages4_ed1_dd8_ddm64':
    #     net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                            with_pos='learned', enc_depth=1, dec_depth=8)
    # elif args.net_G == 'BIT_stages4_ed1_dd8_ddm8':
    #     net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
    #                            with_pos='learned', enc_depth=1, dec_depth=8, decoder_dim_head=8)
    # elif args.net_G == 'ChangeFormerV1':
    #     net = ChangeFormerV1() #ChangeFormer with Transformer Encoder and Convolutional Decoder
    # elif args.net_G == 'ChangeFormerV2':
    #     net = ChangeFormerV2() #ChangeFormer with Transformer Encoder and Convolutional Decoder
    # elif args.net_G == 'ChangeFormerV3':
    #     net = ChangeFormerV3() #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    # elif args.net_G == 'ChangeFormerV4':
    #     net = ChangeFormerV4() #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    # elif args.net_G == 'ChangeFormerV5':
    #     net = ChangeFormerV5(embed_dim=64) #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    # elif args.net_G == 'ChangeFormerV6':
    #     net = ChangeFormerV6(embed_dim=64) #ChangeFormer with Transformer Encoder and Convolutional Decoder (Fuse)
    # elif args.net_G == "DTCDSCN":
    #     #The implementation of the paper"Building Change Detection for Remote Sensing Images Using a Dual Task Constrained Deep Siamese Convolutional Network Model "
    #     #Code copied from: https://github.com/fitzpchao/DTCDSCN
    #     net = CDNet34(in_channels=3)
    # elif args.net_G == "SNUNet_ECAM":
    #     #The implementation of the paper"SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images "
    #     #Code copied from: https://github.com/likyoo/Siam-NestedUNet
    #     net = SNUNet_ECAM(in_ch=3, out_ch=2)
    # elif args.net_G == "IFN":
    #     #A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images
    #     #Code copied from: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images
    #     net = DSIFN()
    # elif args.net_G == "AMT":
    #     #An attention-based multiscale transformer network for remote sensing image change detection "
    #     #https://github.com/linyiyuan11/AMT_Net
    #     net = AMTNet(img_size=args.img_size)
    # elif args.net_G == "A2Net":
    #     #Lightweight Remote Sensing Change Detection  With Progressive Feature Aggregation and  Supervised Attention "
    #     #https://github.com/guanyuezhen/A2Net
    #     net = A2Net()
    # elif args.net_G == "RCTNet":
    #     #Relating CNN-Transformer Fusion Network for  Remote Sensing Change Detection"
    #     #https://github.com/NUST-Machine-Intelligence-Laboratory/RCTNet
    #     net = RCTNet()


    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)




