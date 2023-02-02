# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import glob
from PIL import Image,ImageOps
import logging as log
from tqdm import tqdm
import random
import pandas as pd
import torch
from lpips import LPIPS
from wisp.trainers import BaseTrainer
from wisp.utils import PerfTimer
from wisp.ops.image import write_png, write_exr
from wisp.ops.image.metrics import psnr, lpips, ssim
from wisp.core import Rays


class MultiviewTrainer(BaseTrainer):

    def pre_epoch(self, epoch):
        """Override pre_epoch to support pruning.
        """
        super().pre_epoch(epoch)   
        
        if self.extra_args["prune_every"] > -1 and epoch > 0 and epoch % self.extra_args["prune_every"] == 0:
            self.pipeline.nef.prune()
            self.init_optimizer()
        

    def init_log_dict(self):
        """Custom log dict.
        """
        self.log_dict['total_loss'] = 0
        self.log_dict['total_iter_count'] = 0
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['image_count'] = 0

    def step(self, epoch, n_iter, data):
        """Implement the optimization over image-space loss.
        """
        self.scene_state.optimization.iteration = n_iter

        timer = PerfTimer(activate=False, show_memory=False)

        # Map to device
        rays = data['rays'].to(self.device).squeeze(0)
        img_gts = data['imgs'].to(self.device).squeeze(0)

        timer.check("map to device")

        self.optimizer.zero_grad(set_to_none=True)
        
        timer.check("zero grad")
            
        loss = 0
        
        if self.extra_args["random_lod"]:
            # Sample from a geometric distribution
            population = [i for i in range(self.pipeline.nef.num_lods)]
            weights = [2**i for i in range(self.pipeline.nef.num_lods)]
            weights = [i/sum(weights) for i in weights]
            lod_idx = random.choices(population, weights)[0]
        else:
            # Sample only the max lod (None is max lod by default)
            lod_idx = None

        with torch.cuda.amp.autocast():
            rb = self.pipeline(rays=rays, lod_idx=lod_idx, channels=["rgb"])
            timer.check("inference")

            # RGB Loss
            #rgb_loss = F.mse_loss(rb.rgb, img_gts, reduction='none')
            rgb_loss = torch.abs(rb.rgb[..., :3] - img_gts[..., :3])
            
            rgb_loss = rgb_loss.mean()
            loss += self.extra_args["rgb_loss"] * rgb_loss
            self.log_dict['rgb_loss'] += rgb_loss.item()
            timer.check("loss")

        self.log_dict['total_loss'] += loss.item()
        self.log_dict['total_iter_count'] += 1
        
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        timer.check("backward and step")
        
    def log_tb(self, epoch):
        log_text = 'EPOCH {}/{}'.format(epoch, self.num_epochs)
        self.log_dict['total_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | total loss: {:>.3E}'.format(self.log_dict['total_loss'])
        self.log_dict['rgb_loss'] /= self.log_dict['total_iter_count']
        log_text += ' | rgb loss: {:>.3E}'.format(self.log_dict['rgb_loss'])
        
        for key in self.log_dict:
            if 'loss' in key:
                self.writer.add_scalar(f'Loss/{key}', self.log_dict[key], epoch)

        log.info(log_text)

        self.pipeline.eval()

    def evaluate_metrics(self, epoch, rays, imgs, lod_idx, img_names, name=None):
        
        ray_os = list(rays.origins)
        ray_ds = list(rays.dirs)
        lpips_model = LPIPS(net='vgg').cuda()

        psnr_total = 0.0
        lpips_total = 0.0
        ssim_total = 0.0
        with torch.no_grad():
            for idx, (img, ray_o, ray_d,img_name) in tqdm(enumerate(zip(imgs, ray_os, ray_ds,img_names))):
                
                #idxが1~14ならpredictしない
                print("-------------------------")
                #sprint("idx",idx)
                print("img_name",img_name)
                print("img_names",img_names)

            
                rays = Rays(ray_o, ray_d, dist_min=rays.dist_min, dist_max=rays.dist_max)
                rays = rays.reshape(-1, 3)
                rays = rays.to('cuda')
                rb = self.renderer.render(self.pipeline, rays, lod_idx=lod_idx)
                rb = rb.reshape(*img.shape[:2], -1)
                
                rb.view = None
                rb.hit = None

                rb.gts = img.cuda()
                rb.err = (rb.gts[...,:3] - rb.rgb[...,:3])**2
                psnr_total += psnr(rb.rgb[...,:3], rb.gts[...,:3])
                lpips_total += lpips(rb.rgb[...,:3], rb.gts[...,:3], lpips_model)
                ssim_total += ssim(rb.rgb[...,:3], rb.gts[...,:3])
                
                exrdict = rb.reshape(*img.shape[:2], -1).cpu().exr_dict()
                
                out_name = f"{idx}"
                if name is not None:
                    out_name += "-" + name

                print("========write png & ext=======")
                #print("img",img.keys())
                print("idx",idx)
                print("name",img_name)
                print("name1",name)
                print("out_name",out_name)
                print("lod_idx",lod_idx)
                print("exrdict.keys",exrdict.keys())
                print("exrdict[alpha]",exrdict["alpha"].shape)
                print("exrdict[depth]",exrdict["depth"].shape)
                print("exrdict[default]",exrdict["default"].shape)
                print("rb.cpu().image().byte().rgb.numpy()",rb.cpu().image().byte().rgb.numpy().shape)
                #assert False
                exr_name = os.path.join(self.valid_log_dir, img_name + "-" + out_name + ".exr")
                png_name = os.path.join(self.valid_log_dir, img_name + "-" + out_name + ".png" )
                #write_exr(exr_name, exrdict) #コメントアウト追加
                write_png((png_name), rb.cpu().image().byte().rgb.numpy())
                print("len(rb.cpu().image().byte().rgb.numpy())",len(rb.cpu().image().byte().rgb.numpy()))
                print(len(img_name))
    
                #self.make_error_map(rb.cpu().image().byte().rgb.numpy(),img_name)

        psnr_total /= len(imgs)
        lpips_total /= len(imgs)
        ssim_total /= len(imgs)
                
        log_text = 'EPOCH {}/{}'.format(epoch, self.num_epochs)
        log_text += ' | {}: {:.2f}'.format(f"{name} PSNR", psnr_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} SSIM", ssim_total)
        log_text += ' | {}: {:.6f}'.format(f"{name} LPIPS", lpips_total)
        log.info(log_text)
 
        return {"psnr" : psnr_total, "lpips": lpips_total, "ssim": ssim_total}

    def validate(self, epoch=0):
        self.pipeline.eval()

        log.info("Beginning validation...")
        
        #data = self.dataset.get_images(split="val", mip=2)
        data= self.dataset.get_images(split="val", mip=2)
        print("-------------------",data.keys())
        print("-------------------",data["cameras"].keys())
        
        imgs = list(data["imgs"])

        img_shape = imgs[0].shape
        log.info(f"Loaded validation dataset with {len(imgs)} images at resolution {img_shape[0]}x{img_shape[1]}")

        self.valid_log_dir = os.path.join(self.log_dir, "val")

        print("self.valid_log_dir",self.valid_log_dir)
        print("self.log_dir",self.log_dir)

        log.info(f"Saving validation result to {self.valid_log_dir}")
        #追加 ディレクトリ作成
        if not os.path.exists(self.valid_log_dir):
            os.makedirs(self.valid_log_dir)
            #os.makedirs("テスト")

        metric_dicts = []
        lods = list(range(self.pipeline.nef.num_lods))[::-1]
        for lod in lods:
            #metric_dicts.append(self.evaluate_metrics(epoch, data["rays"], imgs, lod, f"lod{lod}"))
            if lod==1:
                metric_dicts.append(self.evaluate_metrics(epoch, data["rays"], imgs, lod, data["cameras"].keys(),f"lod{lod}"))
        """
        df = pd.DataFrame(metric_dicts)
        df['lod'] = lods
        df.to_csv(os.path.join(self.valid_log_dir, "lod.csv"))
        """

    def make_error_map(self,predict_img_data,img_name):
        img = Image.open(f"./fox/val/{img_name}.jpg")
        #img = img.convert("L")
        #predict_img_data = ImageOps.grayscale(predict_img_data)
        error = Image.fromarray(predict_img_data-predict_img_data)
        #error = Image.fromarray(predict_img_data-img)


        #gray_error = error.convert('L')
        gray_error = ImageOps.grayscale(error)
        #print("difference",predict_img_data-img)
        #print()
        print()
        map_path = os.path.join(self.valid_log_dir, img_name + "-error" + ".png" )

        gray_error.save(map_path)
        #write_png(map_path, gray_error)
        #print("difference",)
        