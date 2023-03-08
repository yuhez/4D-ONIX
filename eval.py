#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate 3D models
This is used to generate a model with the size of the original input, e.g., for H x W inputs, this will generate a H x W x Depth volume.

@author: Yuhe Zhang
"""
import time
import torch
from models.eval_options import ParamOptions
from models.trainer import TrainModel

if __name__ == "__main__":
    opt = ParamOptions().parse()
    model = TrainModel(opt)
    model.is_train = False
    model.generate_3D = True
    model.init_model()
    model.load_trained_models(opt.model_path, opt.load_epoch)
    num_input = 2
    if opt.use_time:
        num_input += 1
    now = time.time()
    with torch.no_grad():
        for k, test_data in enumerate(model.test_loader):
            if k == 0:
                model.set_input(test_data[:num_input])
                model.validation()
                model.visual_iter(opt.load_epoch - 1, k, test_data[num_input].item())
        now = time.time() - now
        print(f"validation time: {now//60} min {round(now - now//60*60)} s")
