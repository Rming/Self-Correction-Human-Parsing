#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import coremltools as ct

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Self Correction for Human Parsing")

    parser.add_argument("--dataset", type=str, default='lip', choices=['lip', 'atr', 'pascal'])
    parser.add_argument("--model-restore", type=str, default='', help="restore pretrained model parameters.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")
    parser.add_argument("--input-dir", type=str, default='', help="path of input image folder.")
    parser.add_argument("--output-dir", type=str, default='', help="path of output image folder.")
    parser.add_argument("--logits", action='store_true', default=False, help="whether to save the logits.")

    return parser.parse_args()




class WrappedResnet101(nn.Module):
    def __init__(self, num_classes, model_restore, input_size):
        super(WrappedResnet101, self).__init__()
        self.model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
        state_dict = torch.load(model_restore)
        self.model.load_state_dict(state_dict)
        self.model.cuda()
        self.model.eval()

        self.upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        output = self.model(x)
        upsample_output = self.upsample(output[0].unsqueeze(0))
        upsample_output = upsample_output.squeeze()
        upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
        return upsample_output



def main():
    args = get_arguments()

    gpus = [int(i) for i in args.gpu.split(',')]
    assert len(gpus) == 1
    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_path = "./pretrain_model/exp-schp-atr.mlmodel"
    num_classes = dataset_settings[args.dataset]['num_classes']
    input_size = dataset_settings[args.dataset]['input_size']
    label = dataset_settings[args.dataset]['label']
    print("Evaluating total class number {} with {}".format(num_classes, label))

    model = WrappedResnet101(num_classes, args.model_restore, input_size).eval()

    example_input = torch.rand(1, 3, input_size[0], input_size[1]).cuda()
    traced_model = torch.jit.trace(model, example_input)

    model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
                name="image",
                shape=example_input.shape,
                color_layout="BGR",
                channel_first=True,
            )
        ],
    )

    # # Use PIL to load and resize the image to expected size.
    # example_image = Image.open("daisy.jpg").resize((224, 224))

    # # Make a prediction using Core ML.
    # out_dict = model.predict({input_name: example_image})

    model.save(model_path)


    # rename output
    ###################################################
    # get model specification
    mlmodel = ct.models.MLModel(str(model_path))
    spec = mlmodel.get_spec()

    # get list of current output_names
    current_output_names = mlmodel.output_description._fd_spec

    # rename first output in list to new_output_name
    old_name = current_output_names[0].name
    new_name = "fushion"
    ct.utils.rename_feature(
        spec, old_name, new_name, rename_outputs=True
    )

    # overwite existing model spec with new renamed spec
    new_model = ct.models.MLModel(spec)
    new_model.save(model_path)

    # xcode preview metadata
    ###################################################
    # Load the saved model
    mlmodel = ct.models.MLModel(model_path)

    # Add new metadata for preview in Xcode
    labels_json = {"labels": dataset_settings[args.dataset]["label"]}

    mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
    mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)

    mlmodel.save(model_path)


if __name__ == '__main__':
    main()
