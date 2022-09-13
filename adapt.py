#!/usr/bin/env python3

# Python standard library
import os
import json
from time import perf_counter
from PIL import Image

# Public libraries
import torch
import torch.nn.functional as functional

# Local imports
import loaders, loaders.online
from harness import Harness
from losses.segmentation import RemappingScore
from bn_adaptation import BNAdaptation
from arguments import AdaptationArguments


class Trainer(Harness):
    def _init_losses(self, opt):
        pass

    def _init_logging(self, opt):
        self.print_frequency = opt.train_print_frequency
        self.tb_frequency = opt.train_tb_frequency
        self.checkpoint_frequency = opt.train_checkpoint_frequency
        self.window = tuple(map(int, opt.adaptation_window.split(', ')))

    def _init_tensorboard(self, opt):
        pass

    def _init_train_loaders(self, opt):
        print('Loading training dataset metadata:', flush=True)

        # Make sure that only the adaptation loader contains entries
        if opt.adaptation_loaders == '':
            raise Exception('The adaptation loader list needs to have entries')

        adaptation_loaders = list(
            getattr(loaders.online, loader_name)(
                resize_height=opt.adaptation_resize_height,
                resize_width=opt.adaptation_resize_width,
                crop_height=opt.adaptation_crop_height,
                crop_width=opt.adaptation_crop_width,
                batch_size=opt.adaptation_batch_size,
                num_workers=opt.sys_num_workers,
                window=self.window  # the window is defined as an argument
            )
            for loader_name in opt.adaptation_loaders.split(',') if (loader_name != '')
        )

        self.adaptation_loaders = loaders.FixedLengthLoaderList(
            adaptation_loaders,
            opt.adaptation_batches
        )

    def _init_training(self, opt):
        self.model_load = opt.model_load
        self.eval_remaps = opt.segmentation_eval_remaps.split(',')
        self.bn_adaptation = BNAdaptation()
        self.adaptation_bn_momentum = opt.adaptation_bn_momentum

    def _flush_logging(self):
        print('', end='', flush=True)

        for writer in self.writers.values():
            writer.flush()

    def _mt(self, sync=False):  # time counter
        if sync:
            torch.cuda.synchronize()
        return perf_counter()

    def _print_full_IoU(self, scores):
        """prints iou of all classes"""
        for domain in scores:
            print('domain               | remap    |     miou | accuracy')

            for remap in scores[domain]:
                metrics = scores[domain][remap].get_scores()

                miou = metrics['meaniou']
                iou = metrics['iou']
                acc = metrics['meanacc']

                print('mIoU:', flush=True)
                print(f'  {domain:20} | {remap:8} | {miou:8.3f} | {acc:8.3f}', flush=True)
                print('IoU:', flush=True)
                print(iou, flush=True)

    def _run_adaptation(self):
        print(f"Epoch {self.state.epoch}:")

        # dictionaries to accumulate the evaluation results after adaptation
        scores = dict()
        images = dict()

        for batch_idx, batch in enumerate(self.adaptation_loaders):
            # load the model to adapt
            self.state.load(self.model_load, disable_lr_loading=True)

            print(f'Adaption and evaluation on image {batch_idx}:')
            with torch.no_grad():
                with self.state.model_manager.get_eval() as model:
                    model = self.bn_adaptation.process(model, self.adaptation_bn_momentum)
                    domain = batch[0]['domain'][0]

                    if domain not in scores:
                        scores[domain] = RemappingScore(self.eval_remaps)
                        images[domain] = list()

                    _ = self._validate_batch_segmentation(model, batch, scores[domain], images[domain])

                    images[domain] = images[domain][:0]

        print(f'Final accumulated metrics after adaptation')
        self._print_full_IoU(scores)

    def adapt(self):
        t0_adaption = self._mt()  # initialize time counter

        self._run_adaptation()

        t1_adaption = self._mt()  # end synchronised time counter
        total_time_adaption = t1_adaption - t0_adaption
        print(f' Total time for adaption & validation: {total_time_adaption:.2f}s')

        print('Completed adaptation without errors', flush=True)
        self._log_gpu_memory()


if __name__ == "__main__":
    opt = AdaptationArguments().parse()

    if opt.sys_best_effort_determinism:
        import numpy as np
        import random

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        random.seed(1)

    trainer = Trainer(opt)
    trainer.adapt()
