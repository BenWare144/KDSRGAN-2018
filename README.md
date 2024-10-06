## KDSRGAN
An application of knowledge distillation in GAN SuperResolution

This code uses an implimentation of the SRGAN network, modified from the code of this repository [SRGAN](https://github.com/tensorlayer/srgan).

### SRGAN Architecture
TensorFlow Implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)

Downloaded pretrained VGG layer [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs).

Download both HR and a set of LR images of this datset:
[DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/) 

This code uses the "unknown downscaling x4".


### Run
Set your image folder and other options in `config.py`.


- Train a network without knowledge distillation
```bash
python main.py
```

- Start evaluation.
```bash
python main.py --mode=evaluate 
```


- Train a network with knowledge distillation.
```bash
python train_distill.py
```

### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

### Author
- [BenWare144](https://github.com/BenWare144)

### License
- For academic and non-commercial use only.
- For commercial use, please contact bwarex@gmail.com.
