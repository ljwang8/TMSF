import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from misc.visualize_results_PIL import visualize_change_detection
from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from misc.utils import de_norm
from misc import utils
from thop import profile


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader
        self.img_size = args.img_size
        self.n_class = args.n_class
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        #定义测试的日志路径
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)

        self.calculate_flops = args.calculate_flops
        #training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir
        self.is_TMSF=args.is_TMSF

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        image_names = self.batch['name']

        for idx, (single_pred, single_gt) in enumerate(zip(self._visualize_pred(), self.batch['L'])):
            single_gt = single_gt * 255
            # 保存预测图像
            original_image_name = image_names[idx]
            single_pred_image = utils.make_numpy_grid(single_pred)
            single_pred_image = np.clip(single_pred_image, 0, 255).astype(np.uint8)
            pred_file_name = os.path.join(self.pred_dir, original_image_name)
            Image.fromarray(single_pred_image).save(pred_file_name)

            # 保存真值标签图像
            single_gt_image = utils.make_numpy_grid(single_gt) # 如果需要，对真值进行处理
            single_gt_image = np.clip(single_gt_image, 0, 255).astype(np.uint8)

            gt_file_name = os.path.join(self.gt_dir, original_image_name)
            Image.fromarray(single_gt_image).save(gt_file_name)

            # Call the visualization function after saving all prediction and ground truth images
            self._create_confusion_matrix_image(single_pred_image, single_gt_image, os.path.join(self.confuse_matrix, original_image_name))


    def _create_confusion_matrix_image(self, pred, gt, output_path):
        pred_gray = np.all(pred == 255, axis=-1).astype(np.uint8)
        gt_gray = np.all(gt == 255, axis=-1).astype(np.uint8)

        TP = (pred_gray == 1) & (gt_gray == 1)
        TN = (pred_gray == 0) & (gt_gray == 0)
        FP = (pred_gray == 1) & (gt_gray == 0)
        FN = (pred_gray == 0) & (gt_gray == 1)

        confusion_image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        confusion_image[TP] = (255, 255, 255)  # 白色
        confusion_image[TN] = (0, 0, 0)  # 黑色
        confusion_image[FP] = (255, 0, 0)  # 红色
        confusion_image[FN] = (0, 255, 0)  # 绿色

        Image.fromarray(confusion_image).save(output_path)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)

        if self.is_TMSF == True:
            name = batch['name']
            self.G_pred = self.net_G(img1=img_in1, img2=img_in2, name=name)[-1]
        else:
            self.G_pred = self.net_G(img_in1, img_in2)[-1]

    def calculate_flops_and_params(self):
        input_shape = (1, 3, self.img_size, self.img_size)
        device = next(self.net_G.parameters()).device
        dummy_input1 = torch.randn(*input_shape).to(device)
        dummy_input2 = torch.randn(*input_shape).to(device)

        flops, params = profile(self.net_G, inputs=(dummy_input1, dummy_input2))
        flops_in_g = flops / 1e9  # GigaFLOPs
        params_in_m = params / 1e6  # MegaParams

        self.logger.write(f"FLOPs: {flops} ({flops_in_g:.2f}G), Params: {params} ({params_in_m:.2f}M)\n")



    def eval_models(self,checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)

        self.pred_dir = os.path.join(self.vis_dir, 'pred')
        os.makedirs(self.pred_dir, exist_ok=True)

        self.gt_dir = os.path.join(self.vis_dir, 'gt')
        os.makedirs(self.gt_dir, exist_ok=True)

        self.confuse_matrix = os.path.join(self.vis_dir, 'confuse_matrix')
        os.makedirs(self.confuse_matrix, exist_ok=True)
        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        if self.calculate_flops == True:
            self.calculate_flops_and_params()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
