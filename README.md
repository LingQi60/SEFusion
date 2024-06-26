# SEFusion
An end-to-end infrared and visible light enhancement fusion algorithm based on SwinTransformer

# To Train
Download the training dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/trainsets/MSRS/**. 

    python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 main.py --opt options/train_vif.json  --dist True

# To Test
Download the test dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/testsets/MSRS/**. 

    python test.py --model_path=./Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/ --dataset=MSRS --A_dir=ir  --B_dir=vi
