# SEFusion
An end-to-end single-stage infrared and visible light enhancement fusion algorithm based on SwinTransformer

## Framework
<img src="https://github.com/LingQi60/SEFusion/blob/main/model/Figs/SEFusion.png" width="989" height="378" /><br/>
The overall framework of the propose SEFusion. It is a U-shaped codec architecture.

## Encoder 
<img src="https://github.com/LingQi60/SEFusion/blob/main/model/Figs/Encoder.png" width="659" height="299" /><br/>
The framework of the decoder.The visible light features pass through the Feaex，TPM and ACM modules in turn.Finally,it is fused with the infrared feature in the CGFM.

## CGFM
<img src="https://github.com/LingQi60/SEFusion/blob/main/model/Figs/CGFM.png" width="671" height="224" /><br/>
The framework of Contrast-guided feature fusion module

## Decoder
<img src="https://github.com/LingQi60/SEFusion/blob/main/model/Figs/Decoder.png" width="663" height="160" /><br/>
The framework of the decoder.The fusion image was reconstructed using a texture-guided self-attention mechanism

## Recommended Environment

-- python 3.8  
-- torch 1.11.0  
-- torchvision 0.12.0  
-- tensorboard  2.7.0  
-- numpy 1.21.2  

## To Train
Download the training dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/trainsets/MSRS/**. 

    python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 main.py --opt options/train_vif.json  --dist True

## To Test
Download the test dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/testsets/MSRS/**. 

    python test.py --model_path=./Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/ --dataset=MSRS --A_dir=ir  --B_dir=vi
    
## Visual Comparison On The MSRS Dataset And TNO Dataset
<img src="https://github.com/LingQi60/SEFusion/blob/main/model/Figs/01.jpg" dth="700" height="480" /><br/>
<img src="https://github.com/LingQi60/SEFusion/blob/main/model/Figs/02.jpg" width="700" height="455" /><br/>

## Result Detection Performance For Images From The LLVIP Dataset 
<img src="https://github.com/LingQi60/SEFusion/blob/main/model/Figs/04.jpg" width="703" height="484" /><br/>

## Acknowledgement
The codes are based on [SwinIR](https://github.com/JingyunLiang/SwinIR). Please also follow their licenses.
