# Recurrent Appearance Flow for Occlusion-Free Virtual Try-On (OF-VTON)

## Introduction
Image-based virtual try-on aims at transferring a target in-shop
garment onto a reference person, which has garnered significant
attention from the research communities recently. However, previous methods have faced severe challenges in handling occlusion
problems. To address this limitation, we classify occlusion problems into three types based on the reference person’s arm postures:
single-arm occlusion, two-arm non-crossed occlusion, and two-arm
crossed occlusion. Specifically, we propose a novel Occlusion-Free
Virtual Try-On Network (OF-VTON) that effectively overcomes
these occlusion challenges. The OF-VTON framework consists of
two core components: i) a new Recurrent Appearance Flow based Deformation (RAFD) model that robustly aligns the in-shop garment
to the reference person by adopting a multi-task learning strategy.
This model jointly produces the dense appearance flow to warp
the garment and predicts a human segmentation map to provide semantic guidance for the subsequent image synthesis model. ii) a
powerful Multi-mask Image SynthesiS (MISS) model that generates
photo-realistic try-on results by introducing a new mask generation
and selection mechanism. Experimental results demonstrate that our
proposed OF-VTON significantly outperforms existing state-of-theart methods by mitigating the impact of occlusion problems.

![baseline](asserts/tryon_baseline.png)

## Environment
* pytorch(1.4.0)
* torchvision
* scipy
* Pillow
* einops
* opencv-python

## Getting Started
### Data Preperation
We provide our **dataset files** , **extracted keypoints files** and **extracted parsing files**  for convience.

[Data Preperation](https://drive.google.com/drive/folders/1VH-i6CZ0AdycHT_rSW0tE2xJVx1rCHYh?usp=share_link)

## Pretrained models
Download the models below and put it under `checkpoint/`

[OFVTON-checkpoints](https://drive.google.com/file/d/1mg8ogkEee3u1WeFJEfmuJkPqQ9pQqWUT/view?usp=share_link)

## Test the model
`python test.py --name test1027 --warp_checkpoint /data2/zjk/PF/checkpoints/train_GMA_refine_4cmaskStage1e2e1xSmoothnoD_holehandAgnostic3AttenmaskUnet_2xmask_avgcontentL_AdamW_bs8_iter5/warp_from_Unet.pth --gen_checkpoint /data2/zjk/PF/checkpoints/train_GMA_refine_4cmaskStage1e2e1xSmoothnoD_holehandAgnostic3AttenmaskUnet_2xmask_avgcontentL_AdamW_bs8_iter5/PBAFN_gen_epoch_101.pth  --dataroot `