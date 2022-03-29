# SSLSOD
<p align="center">
  <img src="./image/logo.png" alt="Logo" width="150" height="auto">


  <h3 align="center">Self-Supervised Pretraining for RGB-D Salient Object Detection</h3>

  <p align="center">
    Xiaoqi Zhao, Youwei Pang, Lihe Zhang, Huchuan Lu, Xiang Ruan
    <br />
    <a href="https://arxiv.org/abs/2101.12482"><strong>⭐ arXiv »</strong></a>
    <a href="https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247591348&idx=2&sn=34a95dbbd14b7adfd508f12899ac27a2&chksm=ec1d884ddb6a015b71612bf16227bed8c69db4074585f712c1dc7dceb7859641088441ff9753&mpshare=1&scene=1&srcid=0112O4ivFbyChzTTH4vKU91t&sharer_sharetime=1646716025907&sharer_shareid=0ffc6ac03af605267e92344350efdb83&exportkey=AxNWDL6LKSsLK6MxSNkSF88%3D&acctmode=0&pass_ticket=TXKuWY6yeluRhUKTt0pk10ycuy%2BMsyJV6%2BXdxFjTtusuYyJMVPywg38icEXhKktM&wx_header=0#rd"><strong>:fire:[Slide&极市平台推送]</strong></a>
    <br /> 
  </p>
</p>

The official repo of the AAAI 2022 paper, Self-Supervised Pretraining for RGB-D Salient Object Detection.
## Saliency map
[Google Drive](https://drive.google.com/file/d/1i5OElgml76p76N2l9eYlFc4Bm8jOlxUk/view?usp=sharing) / [BaiduYunPan(d63j)](https://pan.baidu.com/s/1qifMM7wgR5gPhb6ZlRU9Zw) 
## Trained Model
You can download all trained models at [Google Drive](https://drive.google.com/file/d/1mxX4yk6yOCTapJ_dn_5nZhnb8IpvzEt0/view?usp=sharing) / [BaiduYunPan(0401)](链接: https://pan.baidu.com/s/1zruPGxeR-7j4bfNSrzU5Lg).  
## Datasets
* [Google Drive](https://drive.google.com/file/d/1khN0hTjQ57d5zSFpxADawmeKJ_iQHptu/view?usp=sharing) / [BaiduYunPan(83mj)](https://pan.baidu.com/s/1SfstQCHv0gPV-P3jf4ovjg)  
*  We believe that using a large amount of RGB-D data for pre-training, we will get a super-strong SSL-based model even surpassing the ImageNet-based model. This [survey](https://arxiv.org/pdf/2201.05761.pdf) of the RGB-D dataset may be helpful to you.
## Training
* SSL-based model  
1.Run train_stage1_pretext1.py  
2.Run get_contour.py (can generate the depth-contour maps for the stage2 training)
2.Load the pretext1 weights for Crossmodal_Autoendoer (model_stage1.py) and  run train_stage2_pretext2.py  
3.Load the pretext1 and pretext2 weights for RGBD_sal (model_stage3.py) as initialization and  run train_stage3_downstream.py  
* ImageNet-based model  
Set 'pretrained= Ture' for models.vgg16_bn(pretrained='True') in RGBD_sal (model_stage3.py) and run train_stage3_downstream.py  
## Testing  
Run prediction_rgbd.py (can generate the predicted saliency maps)  
Run test_score.py (can evaluate the predicted saliency maps in terms of fmax,fmean,wfm,sm,em,mae,mdice,miou,ber,acc). 
## BibTex
```
@inproceedings{SSLSOD,
  title={Self-Supervised Pretraining for RGB-D Salient Object Detection},
  author={Zhao, Xiaoqi and Pang, Youwei and Zhang, Lihe and  and Lu, Huchuan and Ruan, Xiang},
  booktitle={AAAI},
  year={2022}
}
```
