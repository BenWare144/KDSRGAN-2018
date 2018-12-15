## KDSRGAN
An application of knowledge distillation in GAN SuperResolution
This code is a modificaiton from the code of this repository [SRGAN](https://github.com/tensorlayer/srgan) which will soon be moved into [here](https://github.com/tensorlayer/tensorlayer/tree/master/examples)

this is an implimentation of the SRGAN network
### SRGAN Architecture
TensorFlow Implementation of ["Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802)

Downloaded pretrained VGG layer[here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

And any images of size 384x384 or higher can be readily used in this dataset:
[DIV2K - bicubic downscaling x4 competition](http://www.vision.ee.ethz.ch/ntire17/) 
download both HR and a set of LR images, I trained using the unknown downscaling x4.


### Run
Set your image folder in `config.py`, Most of the options are there.


-train a network without knowledge distillation
```bash
python train_distill.py
```

- Start evaluation.
```bash
python main.py --mode=evaluate 
```


- Start training.
```bash
python main.py
```

### Reference
* [1] [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

### Author
- [BenWare144](https://github.com/BenWare144)

### License
- For academic and non-commercial use only.
- For commercial use, please contact bwarex@gmail.com.
