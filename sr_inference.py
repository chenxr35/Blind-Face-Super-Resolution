import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference
from mmedit.utils import modify_args, tensor2img

TEST_SIZE = 400


def parse_args():
    modify_args()
    parser = argparse.ArgumentParser(description='Super resolution inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_dir', help='directory of input image files')
    parser.add_argument('save_dir', help='directory of save restoration results')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.device < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', args.device)

    model = init_model(args.config, args.checkpoint, device=device)

    for i in range(TEST_SIZE):
        num = str(i)
        length = len(num)
        order = (5 - length) * '0' + num

        img_path = f'{args.img_dir}/{order}.png'

        # SISR
        output = restoration_inference(model, img_path)

        output = tensor2img(output)

        os.makedirs(args.save_dir, exist_ok=True)

        save_path = f'{args.save_dir}/{order}.png'

        mmcv.imwrite(output, save_path)


if __name__ == '__main__':
    main()
