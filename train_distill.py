#! /usr/bin/python
# -*- coding: utf8 -*-

if True:
    import os, time, pickle, random, time
    from datetime import datetime
    import numpy as np
    from time import localtime, strftime
    import logging, scipy

    import tensorflow as tf
    import tensorlayer as tl
    from model import SRGAN_g, SRGAN_d, Vgg19_simple_api, SRGAN_d_teacher, SRGAN_g_teacher
    from utils import *
    from config import config, log_config
    from main import *
    import os.path

    import skimage

    ### patch warnings ###
    import imageio.core.util
    def silence_imageio_warning(*args, **kwargs):
        pass
    imageio.core.util._precision_warn = silence_imageio_warning #imports

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every #HYPER-PARAMETERS
## knowledge distillation
small_teacher =config.TRAIN.small_techer
train_all_nine =config.TRAIN.train_all_nine
ni = int(np.sqrt(batch_size))

print("starting")
def train_distil():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/student_ginit"
    save_dir_gan = "samples/student_gan"
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    d_losses,g_losses,m_losses,v_losses,a_losses=[],[],[],[],[]
    g0losses,d1losses,d2losses=[],[],[]
    ###====================== PRE-LOAD IMAGE DATA ===========================###

    print("loading images")
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))

    # train_hr_img_list = train_hr_img_list[0:16]
    # train_lr_img_list = train_lr_img_list[0:16]

    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = tl.vis.read_images(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)

    print("images loaded")

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, 96, 96, 3], name='t_image')
    t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')
    # t_distilled_d = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')
    # t_distilled_g = tf.placeholder('float32', [batch_size, 384, 384, 3], name='t_target_image')

    #nets
    net_g_student, net_g_student_distil = SRGAN_g_student(t_image, is_train=True, reuse=False)
    net_d_student, logits_real_student, net_d_student_distil  = SRGAN_d_student(t_target_image, is_train=True, reuse=False)
    net_d_student_fake, logits_fake_student, net_d_student_distil_fake = SRGAN_d_student(net_g_student.outputs, is_train=True, reuse=True)



    if small_techer is True and train_all_nine is False:
        net_g_teacher_distil = SRGAN_g_teacher_small(t_image, is_train=False, reuse=False)
        net_d_teacher_distil  = SRGAN_d_teacher_small(t_target_image, is_train=False, reuse=False)
        net_d_teacher_distil_fake = SRGAN_d_teacher_small(net_g_student.outputs, is_train=False, reuse=True)
    else:
        net_g_teacher, net_g_teacher_distil = SRGAN_g_teacher(t_image, is_train=False, reuse=False)
        net_d_teacher, _, net_d_teacher_distil  = SRGAN_d_teacher(t_target_image, is_train=False, reuse=False)
        net_d_teacher_fake, _, net_d_teacher_distil_fake = SRGAN_d_teacher(net_g_student.outputs, is_train=False, reuse=True)

    if not train_all_nine is True:
        net_g0_predict, _ = SRGAN_g0_predict(net_g_student_distil.outputs, is_train=True, reuse=False)
        net_d1_predict, _ = SRGAN_d1_predict(net_d_student_distil.outputs, is_train=True, reuse=False)
        net_d2_predict, _ = SRGAN_d2_predict(net_d_student_distil_fake.outputs, is_train=True, reuse=False)
    else:

        net_g0_predict, _ = SRGAN_g0_predict(net_g_student_distil.outputs, is_train=True, reuse=False)
        net_d1_predict, _ = SRGAN_d1_predict(net_d_student_distil.outputs, is_train=True, reuse=False)
        net_d2_predict, _ = SRGAN_d2_predict(net_d_student_distil_fake.outputs, is_train=True, reuse=False)

        net_d1d2_predict, _ = SRGAN_d1d2_predict(net_d_student_distil.outputs, is_train=True, reuse=False)
        net_d2d1_predict, _ = SRGAN_d2d1_predict(net_d_student_distil_fake.outputs, is_train=True, reuse=False)

        net_g0d1_predict, _ = SRGAN_g0d1_predict(net_g_student_distil.outputs, is_train=True, reuse=False)
        net_g0d2_predict, _ = SRGAN_g0d2_predict(net_g_student_distil.outputs, is_train=True, reuse=False)

        net_d1g0_predict, _ = SRGAN_d1g0_predict(net_d_student_distil.outputs, is_train=True, reuse=False)
        net_d2g0_predict, _ = SRGAN_d2g0_predict(net_d_student_distil.outputs, is_train=True, reuse=False)




    net_g_student.print_params(False)
    net_g_student.print_layers()
    net_d_student.print_params(False)
    net_d_student.print_layers()
    # net_g_student_distil_fake.print_params(False)
    # net_g_student_distil_fake.print_layers()
    # net_d_student_distil_fake.print_params(False)
    # net_d_student_distil_fake.print_layers()

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g_student.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)
    ## test inference
    net_g_test, _ = SRGAN_g_student(t_image, is_train=True, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real_student, tf.ones_like(logits_real_student), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake_student, tf.zeros_like(logits_fake_student), name='d2')
    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake_student, tf.ones_like(logits_fake_student), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g_student.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    if not train_all_nine is True:
        g0_loss = 4e-3 *tl.cost.mean_squared_error(net_g0_predict.outputs, net_g_teacher_distil.outputs, is_mean=True)
        d1_loss = 1/5*tl.cost.mean_squared_error(net_d1_predict.outputs, net_d_teacher_distil.outputs, is_mean=True)
        d2_loss = 1/5*tl.cost.mean_squared_error(net_d2_predict.outputs, net_d_teacher_distil_fake.outputs, is_mean=True)
    else:
        g0_loss = 4e-3 *tl.cost.mean_squared_error((net_g0_predict.outputs + net_d1g0_predict.outputs + net_d2g0_predict.outputs)/3, net_g_teacher_distil.outputs, is_mean=True)
        d1_loss = 1/5*tl.cost.mean_squared_error((net_g0d1_predict.outputs + net_d1_predict.outputs + net_d2d1_predict.outputs)/3, net_d_teacher_distil.outputs, is_mean=True)
        d2_loss = 1/5*tl.cost.mean_squared_error((net_g0d2_predict.outputs + net_d1d2_predict.outputs + net_d2_predict.outputs)/3, net_d_teacher_distil_fake.outputs, is_mean=True)



    d_loss = d_loss1 + d_loss2 + d1_loss + d2_loss
    g_loss = mse_loss + vgg_loss + g_gan_loss + g0_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g_student', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d_student', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    # ## Pretrain
    # g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)



    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_student.npz', network=net_g_student) is False:
        pass
        # tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_init.npz', network=net_g_student)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_srgan_student.npz', network=net_d_student)

    if small_teacher is True:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_small_teacher_bicube.npz', network=net_g_teacher_distil)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_small_teacher_bicube.npz', network=net_d_teacher_distil)
    else:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_teacher.npz', network=net_g_teacher)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_srgan_teacher.npz', network=net_d_teacher)
    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
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
    ## use first `batch_size` of train set to have a quick test during training
    # sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = train_lr_imgs[0:batch_size]
    # # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384, sample_imgs_96 = threading_data_2((train_hr_imgs[0:batch_size],
                                                        train_lr_imgs[0:batch_size]), fn=crop2, is_random=True)
    # sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=True)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    # sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit + '/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan + '/_train_sample_384.png')
    ###========================= train GAN (SRGAN) =========================###
    print("starting")
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0
        total_errM, total_errV, total_errA = 0,0,0
        total_erg0, total_erd1, total_erd2 = 0,0,0

        ## Actual training
        for idx in range(0, len(train_hr_imgs), batch_size):
                #using 4x lowresolution images
            b_imgs_384, b_imgs_96 = threading_data_2((train_hr_imgs[idx:idx + batch_size],
                                    train_lr_imgs[idx:idx + batch_size]), fn=crop2, is_random=True)
                #for 4x high resolution images
            # if n_iter==0 and epoch==0: tl.vis.save_images(b_imgs_384, [ni, ni], save_dir_gan + '/original_train_384_%d.png' % epoch)
            # if n_iter==0 and epoch==0: tl.vis.save_images(b_imgs_96, [ni, ni], save_dir_gan + '/original_train_96_%d.png' % epoch)
            step_time = time.time()
                ## update D
            errD, erg0, erd1, erd2, _ = sess.run([d_loss, g0_loss, d1_loss, d2_loss, d_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})
                ## update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})

            total_d_loss += errD
            total_g_loss += errG
            total_errM += errM
            total_errV += errV
            total_errA += errA
            total_erg0 += erg0
            total_erd1 += erd1
            total_erd2 += erd2
            n_iter += 1
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)(erg0: %.6f erd1: %.6f erd2: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA, erg0, erd1, erd2))
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        ##write losses
        d_losses += [total_d_loss / n_iter]
        g_losses += [total_g_loss / n_iter]
        m_losses += [total_errM   / n_iter]
        v_losses += [total_errV   / n_iter]
        a_losses += [total_errA   / n_iter]
        g0losses += [total_erg0   / n_iter]
        d1losses += [total_erd1   / n_iter]
        d2losses += [total_erd2   / n_iter]

        if epoch % 20 == 0:
            tl.files.save_npz(net_g_student.all_params, name=checkpoint_dir + '/g_srgan_student_%d.npz'%epoch, sess=sess)
            tl.files.save_npz(net_d_student.all_params, name=checkpoint_dir + '/d_srgan_student_%d.npz'%epoch, sess=sess)
            write_losses("d_losses",d_losses)
            write_losses("g_losses",g_losses)
            write_losses("m_losses",m_losses)
            write_losses("v_losses",v_losses)
            write_losses("a_losses",a_losses)
            write_losses("g0losses",g0losses)
            write_losses("d1losses",d1losses)
            write_losses("d2losses",d2losses)

        if epoch % 10 == 0:
            # out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            tl.files.save_npz(net_g_student.all_params, name=checkpoint_dir + '/g_srgan_student.npz', sess=sess)
            tl.files.save_npz(net_d_student.all_params, name=checkpoint_dir + '/d_srgan_student.npz', sess=sess)
            if not small_teacher is True:
                tl.files.save_npz(net_g_teacher_distil.all_params, name=checkpoint_dir + '/g_small_teacher_bicube.npz', sess=sess)
                tl.files.save_npz(net_d_teacher_distil.all_params, name=checkpoint_dir + '/d_small_teacher_bicube.npz', sess=sess)
            evaluate()

        ## quick evaluation on train set
        if  (epoch % 5 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')
    args = parser.parse_args()
    # tl.global_flag['mode'] = args.mode
    tl.global_flag['mode'] = 'srgan'
    train_distil()
