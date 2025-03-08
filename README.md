# CellResDM  
This repository contains the implementation of our paper:  
**Artificial immunofluorescence in a flash: Rapid synthetic imaging from brightfield through residual diffusion.**  

## Acknowledgement  
This repository is heavily inspired by [ResShift](https://github.com/zsyOAOA/ResShift), which introduced the use of residual diffusion for super-resolution models. Our implementation adapts this method for the **five-channel Cell Painting dataset**, with several key modifications:  

1. **Custom TIFF Dataset Implementation**  
   - We introduced a new dataset module, `dataset_tiff.py`, under `basicsr/data`.  
   - This module handles preprocessing for TIFF images used in our study.  
   - Corresponding configuration updates were made throughout the repository to ensure seamless dataset loading.  

2. **VQ-VAE Model for Cell Painting Data**  
   - We trained an additional VQ-VAE model specifically for the five-channel Cell Painting dataset.  
   - Checkpoints can be found at: **[insert link]**.  
   - However, based on our experiments, we recommend skipping the VQ-VAE step and using a **512×512 resolution** instead.  

3. **New Configuration Files**  
   - We added a new YAML configuration file under the `configs/` folder.  
   - This file contains the exact settings used in our experiments.  

## Contributions  
Our paper makes three major contributions:  

1. **Comparative Analysis in Cell Painting Tasks**  
   - We benchmarked this method against other models for Cell Painting tasks.  
   - The code and checkpoints for the compared models are available **upon request** (as they are still being organized).  

2. **Hybrid Synthesis of Segmentation Masks & Images**  
   - We demonstrate how diffusion models can synthesize both segmentation masks and images.  

3. **State-of-the-Art Performance**  
   - This implementation achieves the best performance in our comparisons.  
   - Checkpoints can be found at: **[insert link]**.  


## How to Use  

### Inference  
To run inference, use the following command:  
```bash
python predict.py
```  
Make sure to update the paths to the **saved checkpoints** before running the script.  

### Training  

#### Training Without VAE  
To train a custom model **without VAE**, use:  
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 --nnodes=1 main.py \
  --cfg_path configs/custom_noae.yaml --save_dir [Logging Folder]
```

#### Training With VAE (Pretrained on Cell Painting Dataset)  
To train a custom model **with VAE**, use:  
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 --nnodes=1 main.py \
  --cfg_path configs/custom_5channelae.yaml --save_dir [Logging Folder]
```




