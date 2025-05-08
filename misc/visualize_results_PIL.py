import os
import os.path as osp
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

class CDVisualization(object):
    def __init__(self, policy=['compare_pixel', 'pixel']):
        """Change Detection Visualization
        Args:
            policy (list, optional): Visualization policies. Defaults to ['compare_pixel', 'pixel'].
        """
        super().__init__()
        self.policy = policy
        self.num_classes = 2
        self.COLOR_MAP = {
            '0': (0, 0, 0),  # Black for True Negative (TN)
            '1': (0, 255, 0),  # Green for False Positive (FP)
            '2': (255, 0, 0),  # Red for False Negative (FN)
            '3': (255, 255, 255)  # White for True Positive (TP)
        }

    def read_and_check_label(self, file_name):
        try:
            img = Image.open(file_name).convert('L')  # Convert to grayscale
            img = np.array(img)
            img[img < 128] = 0
            img[img >= 128] = 1
            return img
        except IOError:
            raise FileNotFoundError(f"Failed to read the image {file_name}. Check if the file is corrupted or in an unsupported format.")

    def read_img(self, file_name):
        try:
            img = Image.open(file_name)  # Default reads in RGB
            return np.array(img)
        except IOError:
            raise FileNotFoundError(f"Failed to read the image {file_name}. Check if the file is corrupted or in an unsupported format.")

    def save_img(self, file_name, vis_res, imgs=None):
        vis_res = Image.fromarray(vis_res)
        vis_res.save(file_name)

    def trainIdToColor(self, trainId):
        return self.COLOR_MAP[str(trainId)]

    def gray2color(self, grayImage, num_class):
        rgbImage = np.zeros((grayImage.shape[0], grayImage.shape[1], 3), dtype=np.uint8)
        for cls in num_class:
            row, col = np.where(grayImage == cls)
            if len(row) > 0:
                rgbImage[row, col] = self.trainIdToColor(cls)
        return rgbImage

    def res_pixel_visual(self, label):
        label_rgb = self.gray2color(label, num_class=list(range(self.num_classes)))
        return label_rgb

    def res_compare_pixel_visual(self, pred, gt):
        visual = self.num_classes * gt.astype(int) + pred.astype(int)
        visual_rgb = self.gray2color(visual, num_class=list(range(self.num_classes ** 2)))
        return visual_rgb

    def __call__(self, pred_path, gt_path, dst_path, imgs=None):
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        os.makedirs(dst_path)

        pred = self.read_and_check_label(pred_path)
        gt = self.read_and_check_label(gt_path)

        for pol in self.policy:
            dst_path_pol = osp.join(dst_path, pol)
            os.makedirs(dst_path_pol, exist_ok=True)
            dst_file = osp.join(dst_path_pol, osp.basename(dst_path))

            if pol == 'compare_pixel':
                visual_map = self.res_compare_pixel_visual(pred, gt)
                self.save_img(dst_file, visual_map, None)
            elif pol == 'pixel':
                visual_map = self.res_pixel_visual(pred)
                self.save_img(dst_file, visual_map, imgs)
            else:
                raise ValueError(f"Invalid policy {pol}")


def visualize_change_detection(gt_dir, pred_dir, dst_dir):

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    file_name_list = os.listdir(gt_dir)

    for file_name in file_name_list:
        gt_path = os.path.join(gt_dir, file_name)
        pred_path = os.path.join(pred_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        visualizer = CDVisualization(policy=['compare_pixel'])
        visualizer(pred_path, gt_path, dst_path)

