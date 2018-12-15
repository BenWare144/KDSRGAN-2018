#! /usr/bin/python
# -*- coding: utf8 -*-
if True:  # preamble
    import os
    import time
    import pickle
    import random
    import time
    from datetime import datetime
    import numpy as np
    from time import localtime, strftime
    import logging
    import scipy

    import tensorflow as tf
    import tensorlayer as tl
    from model import *  # SRGAN_g, SRGAN_d, Vgg19_simple_api
    from utils import *
    from config import config, log_config
    import os.path

    import skimage

    ### patch warnings ###
    import imageio.core.util

    def silence_imageio_warning(*args, **kwargs):
        pass
    imageio.core.util._precision_warn = silence_imageio_warning  # imports

###====================== HYPER-PARAMETERS ===========================###
# Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
# initialize G
n_epoch_init = config.TRAIN.n_epoch_init
# adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train():
    # create folders to save result images and trained model

    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    d_losses, g_losses, m_losses, v_losses, a_losses = [], [], [], [], []
    ###====================== PRE-LOAD IMAGE DATA ===========================###

    print("loading images")
    train_hr_img_list = sorted(tl.files.load_file_list(
        path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(
        path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    # shrink training set for debugging.
    # train_hr_img_list = train_hr_img_list[0:16]
    # train_lr_img_list = train_lr_img_list[0:16]

    # If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = tl.vis.read_images(
        train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = tl.vis.read_images(
        train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)

    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    print("images loaded")
    # exit()

    ###========================== DEFINE MODEL ============================###
    # train inference
    t_image = tf.placeholder(
        'float32', [batch_size, 96, 96, 3], name='t_image')
    t_target_image = tf.placeholder(
        'float32', [batch_size, 384, 384, 3], name='t_target_image')

    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    # net_g.print_params(False)
    # net_g.print_layers()
    # net_d.(False)
    # net_d.print_layers()

    # vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0, align_corners=False)
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[
                                                 224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api(
        (t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api(
        (t_predict_image_224 + 1) / 2, reuse=True)

    # test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(
        logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(
        logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-3 * \
        tl.cost.sigmoid_cross_entropy(
            logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(
        net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * \
        tl.cost.mean_squared_error(
            vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    # Pretrain
    g_optim_init = tf.train.AdamOptimizer(
        lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    # SRGAN
    g_optim = tf.train.AdamOptimizer(
        lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(
        lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        pass
        # tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir +
                                 '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print(
            "Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= PreSample ===============================###
    # use first `batch_size` of train set to have a quick test during training
    # sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = train_lr_imgs[0:batch_size]
    # # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set

    sample_imgs_384, sample_imgs_96 = threading_data_2((train_hr_imgs[0:batch_size],
                                                        train_lr_imgs[0:batch_size]), fn=crop2, is_random=True)

    # sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=True)
    # sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample HR sub-image:', sample_imgs_384.shape,
          sample_imgs_384.min(), sample_imgs_384.max())
    print('sample LR sub-image:', sample_imgs_96.shape,
          sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(
        sample_imgs_96, [ni, ni], save_dir_ginit + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [
                       ni, ni], save_dir_ginit + '/_train_sample_384.png')
    tl.vis.save_images(
        sample_imgs_96, [ni, ni], save_dir_gan + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [
                       ni, ni], save_dir_gan + '/_train_sample_384.png')

    # I did not use this function, it seemed to not help very much
    """
    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384, b_imgs_96 = threading_data_2((train_hr_imgs[idx:idx + batch_size],
                                train_lr_imgs[idx:idx + batch_size]), fn=crop2, is_random=True)
            ## update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 4 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 4 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
            evaluate()
    # """

    ###========================= train GAN (SRGAN) =========================###

    for epoch in range(0, n_epoch + 1):
        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (
                lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (
                lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0
        total_errM, total_errV, total_errA = 0, 0, 0

        # If your machine cannot load all images into memory, you should use":
        # this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        print("starting epoch")

        # Actual training
        for idx in range(0, len(train_hr_imgs), batch_size):
            b_imgs_384, b_imgs_96 = threading_data_2((train_hr_imgs[idx:idx + batch_size],
                                                      train_lr_imgs[idx:idx + batch_size]), fn=crop2, is_random=True)
            # if n_iter==0 and epoch==0: tl.vis.save_images(b_imgs_384, [ni, ni], save_dir_gan + '/original_train_384_%d.png' % epoch)
            # if n_iter==0 and epoch==0: tl.vis.save_images(b_imgs_96, [ni, ni], save_dir_gan + '/original_train_96_%d.png' % epoch)
            step_time = time.time()

            # update D
            errD, _ = sess.run([d_loss, d_optim], {
                               t_image: b_imgs_96, t_target_image: b_imgs_384})
            # update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {
                                                 t_image: b_imgs_96, t_target_image: b_imgs_384})

            total_d_loss += errD
            total_g_loss += errG
            total_errM += errM
            total_errV += errV
            total_errA += errA
            n_iter += 1
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        # write losses
        d_losses += [total_d_loss / n_iter]
        g_losses += [total_g_loss / n_iter]
        m_losses += [total_errM / n_iter]
        v_losses += [total_errV / n_iter]
        a_losses += [total_errA / n_iter]

        if epoch % 10 == 0:
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir +
                              '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir +
                              '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            write_losses("d_losses", d_losses)
            write_losses("g_losses", g_losses)
            write_losses("m_losses", m_losses)
            write_losses("v_losses", v_losses)
            write_losses("a_losses", a_losses)
            evaluate()
        if epoch % 20 == 0:
            # out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            tl.files.save_npz(
                net_g.all_params, name=checkpoint_dir + '/g_%d.npz' % epoch, sess=sess)
            tl.files.save_npz(
                net_d.all_params, name=checkpoint_dir + '/d_%d.npz' % epoch, sess=sess)

        # quick evaluation on train set
        if (epoch % 5 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})
            print("[*] save images")
            tl.vis.save_images(
                out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)


def evaluate():
    print("evaluating")
    # create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"
    print("loading images")

    valid_hr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(
        path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    if tl.global_flag['mode'] == 'srgan':
        end = 10  # EVALUATIONS WHILE TRAINING
    elif tl.global_flag['mode'] == 'evaluate':
        end = len(valid_hr_img_list) // 2  # EVALUATIONS ON EVALUATE
    else:
        raise Exception("Unknow --mode")

    ###====================== PRE-LOAD DATA ===========================###
    if tl.global_flag['mode'] == 'single':
        input_file_list = config.Valid.input_file_list
        valid_hr_img_list = sorted(input_file_list)
        valid_lr_img_list = sorted(input_file_list)

        for i in range(len(valid_hr_img_list)):
            if len(valid_lr_imgs[i].shape) != 3:
                print("Image " + str(i) + " old shape: " +
                      str(valid_lr_imgs[i].shape))
                # .reshape(1,valid_lr_imgs[i].shape[1],valid_lr_imgs[i].shape[2],3)
                valid_lr_imgs[i] = skimage.color.gray2rgb(valid_lr_imgs[i])
                print("Image " + str(i) + " new shape: " +
                      str(valid_lr_imgs[i].shape))

    else:  # elif tl.global_flag['mode'] == 'evaluate':
        # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
        # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
        # If your machine have enough memory, please pre-load the whole train set.
        # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
        valid_lr_imgs = tl.vis.read_images(
            valid_lr_img_list[:end], path=config.VALID.lr_img_path, n_threads=32)
        valid_hr_imgs = tl.vis.read_images(
            valid_hr_img_list[:end], path=config.VALID.hr_img_path, n_threads=32)

    print("images loaded")

    ###========================== DEFINE MODEL ============================###
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    # net_g = SRGAN_g(t_image, is_train=False, reuse=True)

    try:
        net_g = SRGAN_g(t_image, is_train=False, reuse=False)
    except:
        net_g = SRGAN_g(t_image, is_train=False, reuse=True)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    #load the highest prority model
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_student.npz', network=net_g):
        print("using /g_student.npz")
    elif tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g):
        print("using /g_srgan.npz")
    elif tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_init.npz', network=net_g):
        print("using /g_srgan_init.npz")
    else:
        print("error! no model to eval")
        # tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_student.npz', network=net_g)

    for imid in range(end):
        start_time = time.time()
        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your ow n image
        valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
        # print(valid_lr_img.min(), valid_lr_img.max())

        size = valid_lr_img.shape

        ###======================= EVALUATION =============================###
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})

        save_name_num = 0
        while True:
            # print(str(save_name_num))
            save_name = save_dir + '/' + \
                str(imid) + '_valid_gen_' + str(save_name_num) + '.png'
            if not os.path.isfile(save_name):
                break
            save_name_num += 1
        tl.vis.save_image(out[0], save_name)

        print("[*] save images" + save_name +
              ("took: %4.4fs" % (time.time() - start_time)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan',
                        help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate' or tl.global_flag['mode'] == 'single':
        evaluate()
    else:
        raise Exception("Unknow --mode")
