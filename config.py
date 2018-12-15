from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
config.TRAIN.lr_decay_init = 0.1
config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)
direct='/home/ben/Documents/Computer_vision/CV_Project/code/'
## train set location
config.TRAIN.hr_img_path = direct + 'image/DIV2K_train_HR/'
config.TRAIN.lr_img_path = direct + 'image/DIV2K_train_LR_unknown/X4/'
config.VALID = edict()
## test set location
config.VALID.hr_img_path = direct + 'image/DIV2K_valid_HR/'
config.VALID.lr_img_path = direct + 'image/DIV2K_valid_LR_unknown/X4/'
## knowledge distillation
config.TRAIN.small_techer = True    # True, loads only the parts of the teacher upto the network.
config.TRAIN.train_all_nine = True  # True trains the three specific student layers to predict the activations of the three teacher layers.
## if gone wants to test the generator on a outside image or set of images
config.VALID.input_file_list=["/home/ben/Documents/Computer_vision/CV_Project/code/CVPR2017_linedrawings/mysamples/output.jpg"]

## warning, model saves every epoch, which consumes ~300MB per save fie

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
