# Example run: python -W ignore SPAQ.py --test_dir /root/autodl-tmp/Result/RetinexNet/ExDark --read_subfolder True

import torch
import torch.nn as nn
import torchvision
# from Prepare_image import Image_load
from PIL import Image
import argparse
import os
import numpy as np
import numpy
from torchvision import transforms
from PIL import Image
import torch
from glob import glob

class Image_load(object):
    def __init__(self, size, stride, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.stride = stride
        self.interpolation = interpolation

    def __call__(self, img):
        image = self.adaptive_resize(img)
        return self.generate_patches(image, input_size=self.stride)
       
    def adaptive_resize_old(self, img):
        pass

    def adaptive_resize(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size
        if h < self.size or w < self.size:
            img = transforms.ToTensor()(img)
            # print('if img.size=', img.size(), type(img))
            return img
        else:
            img = transforms.ToTensor()(transforms.Resize(self.size, self.interpolation)(img))
            # print('else img.size=', img.size(), type(img))
            return img

    def to_numpy(self, image):
        p = image.numpy()
        return p.transpose((1, 2, 0))

    def generate_patches(self, image, input_size, type=np.float32):
        img = self.to_numpy(image)
        img_shape = img.shape
        img = img.astype(dtype=type)
        if len(img_shape) == 2:
            H, W, = img_shape
            ch = 1
        else:
            H, W, ch = img_shape
        if ch == 1:
            img = np.asarray([img, ] * 3, dtype=img.dtype)

        stride = int(input_size / 2)
        hIdxMax = H - input_size
        wIdxMax = W - input_size

        hIdx = [i * stride for i in range(int(hIdxMax / stride) + 1)]
        if H - input_size != hIdx[-1]:
            hIdx.append(H - input_size)
        wIdx = [i * stride for i in range(int(wIdxMax / stride) + 1)]
        if W - input_size != wIdx[-1]:
            wIdx.append(W - input_size)
        patches_numpy = [img[hId:hId + input_size, wId:wId + input_size, :]
                    for hId in hIdx
                    for wId in wIdx]
        patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        # for i in range(len(patches_tensor)):
        #     print(patches_tensor[i].shape)
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()
        return patches_tensor.squeeze(0)


class Baseline(nn.Module):
	def __init__(self):
		super(Baseline, self).__init__()
		self.backbone = torchvision.models.resnet50(pretrained=False)
		fc_feature = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

	def forward(self, x):
		result = self.backbone(x)
		return result

class Demo(object):
	def __init__(self, config, load_weights=True, checkpoint_dir='./weights/BL_release.pt' ):
		self.config = config
		self.load_weights = load_weights
		self.checkpoint_dir = checkpoint_dir

		self.prepare_image = Image_load(size=512, stride=224)

		self.model = Baseline()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)
		self.model_name = type(self.model).__name__

		if self.load_weights:
			self.initialize()

	def run(self):
		res = []
		img_dir = self.config.test_dir

		if self.config.read_subfolder:
			path = glob(os.path.join(img_dir, '*/*'))
		else:
			path = glob(os.path.join(img_dir, '*'))

		# print("img_dir is", img_dir)
		RuntimeError_count = 0
		for i in range(len(path)):
			try:
				score = self.predit_quality(path[i]).item() # NOTE: use .item() to detach gradient
			except RuntimeError:
				RuntimeError_count += 1
				print(i, " path[i] is", path[i])
			res.append(score)
		print("Have ", RuntimeError_count, " times RuntimeError.")
		print("Average SPAQ:", "%.3f" % (sum(res) / len(res)))


	def predit_quality(self, img_path):
		image = self.prepare_image(Image.open(img_path).convert("RGB"))
		image = image.to(self.device)
		self.model.eval()
		score = self.model(image).mean()
		return score

	def initialize(self):
		ckpt_path = self.checkpoint_dir
		could_load = self._load_checkpoint(ckpt_path)
		if could_load:
			print('Checkpoint load successfully!')
		else:
			raise IOError('Fail to load the pretrained model')

	def _load_checkpoint(self, ckpt):
		if os.path.isfile(ckpt):
			print("[*] loading checkpoint '{}'".format(ckpt))
			checkpoint = torch.load(ckpt, map_location = self.device)
			self.model.load_state_dict(checkpoint['state_dict'])
			return True
		else:
			return False

def parse_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--test_dir', type=str)
	parser.add_argument('--read_subfolder', type=bool, default=False)
	return parser.parse_args()

def main():
	cfg = parse_config()
	t = Demo(config=cfg)
	t.run()


# NOTE: the original run function
# def run(self):
	# 	res = []
	# 	img_dir = self.config.test_dir
	# 	# print("img_dir is", img_dir)
	# 	for img_file in os.listdir(img_dir):
	# 		img_path = os.path.join(img_dir, img_file)
	# 		score = self.predit_quality(img_path).item() # NOTE: use .item() to detach gradient
	# 		res.append(score)

	# 	print("Average SPAQ:", "%.3f" % (sum(res) / len(res)))

if __name__ == '__main__':
	main()
