# U-Net##: A Powerful Novel Architecture for Medical Image Segmentation
Official PyTorch implementation of the MICAD 2022 conference paper: ["U-Net##: A Powerful Novel Architecture for Medical Image Segmentation"](https://link.springer.com/chapter/10.1007/978-981-16-6775-6_19)

## Information

In this paper, we propose a powerful novel architecture named U-Net##, which consists of multiple overlapping U-Net pathways and has the strategies of sharing feature maps between parallel neural networks, using auxiliary convolutional blocks for additional feature extractions and deep supervision, so that it performs as a boosted U-Net model for medical image segmentation.

## Architecture

### Block Diagram of the U-Net## Architecture

<img title="U-Net## Block Diagram" src="https://github.com/firatkorkmaz/UNetSharpSharp/blob/main/images/UNetSharpSharp.png">

### Convolutional Blocks of the U-Net## Architecture

<img title="U-Net## Convolutional Blocks" src="https://github.com/firatkorkmaz/UNetSharpSharp/blob/main/images/UNetSharpSharpCB.png">

## Results

The U-Net## model is evaluated on the TCIA-LGG Segmentation Dataset from The Cancer Imaging Archive (TCIA) to segment the brain regions with FLAIR abnormalities on the related brain MRI images.

### Some Output Images Predicted by the Trained Models

<img title="Predicted Output Images" src="https://github.com/firatkorkmaz/UNetSharpSharp/blob/main/images/UNetSharpSharpComparison.png">

### Score Results of the Trained Models

<img title="Score Results" src="https://github.com/firatkorkmaz/UNetSharpSharp/blob/main/images/UNetSharpSharpScores.png">

### Comparison of the Dice Score Changes

<img title="Dice Coefficient per Epoch" src="https://github.com/firatkorkmaz/UNetSharpSharp/blob/main/images/UNetSharpSharpDiceGraph.png">

## Citation

If you find this work useful for your research, please consider citing:

```
@InProceedings{10.1007/978-981-16-6775-6_19,
author={Korkmaz, Fırat},
editor={Su, Ruidan and Zhang, Yudong and Liu, Han and F Frangi, Alejandro},
title={U-Net##: A Powerful Novel Architecture for Medical Image Segmentation},
booktitle={Medical Imaging and Computer-Aided Diagnosis},
year={2023},
publisher={Springer Nature Singapore},
address={Singapore},
pages={231--241},
isbn={978-981-16-6775-6}
}
```
