import os
os.environ["PYOPENGL_PLATFORM"] = "OSMesa"
import torch
from tqdm import tqdm
import cv2
import time
import torchvision.transforms as standard

from base import Tester
from config import cfg
from utils.visualize import *
from data.processing import *

class HandTracker():
    def __init__(self):
        self.tester = Tester()

        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = standard.Compose([standard.ToTensor(), standard.Normalize(*mean_std)])

    def run(self, color):
        inputs, targets = self.tester.get_record(color)
        with torch.no_grad():
            return self.tester.model(inputs)

    def get_record(self, img):
        img = self.transform(img)
        inputs = np.float32(img)
        targets = {}

        image = torch.from_numpy(inputs)
        img = torch.unsqueeze(image, 0).type(torch.float32)
        inputs = {'img': img}
        return inputs, targets


def _get_input(frame):
    ### load image from recorded files ###
    load_filepath = './recorded_files/'

    color = cv2.imread(load_filepath + 'color_%d.png' % frame)
    color = cv2.resize(color, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

    return color

def _visualize(color, outs):
    outs = {k: v.cpu().numpy() for k, v in outs.items()}
    coords_uvd = outs['coords'][0]
    coords_uvd[:, :2] = (coords_uvd[:, :2] + 1) * cfg.input_img_shape[0] // 2

    vis = draw_2d_skeleton(color, coords_uvd[cfg.num_vert:])
    vis = cv2.resize(vis, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    color = cv2.resize(color, dsize=(416, 416), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("vis", vis)
    cv2.imshow("img", color)
    cv2.waitKey(50)


def main():
    torch.backends.cudnn.benchmark = True

    tracker = HandTracker()

    frame = 1
    while True:
        color = _get_input(frame)

        outs = tracker.run(color)

        _visualize(color, outs)

        frame += 1


if __name__ == '__main__':
    main()



