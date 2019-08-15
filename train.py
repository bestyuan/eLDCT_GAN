import time
import math
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from skimage import measure
# from tensorflow.python import debug as tf_debug
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)

def train(data_training,data_groundtruth):
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    gen_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='generated_image')
    real_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='groundtruth_image')
    #add learningrate
    d_learning_rate = tf.placeholder(dtype=tf.float32,name='d_learning_rate')
    g_learning_rate = tf.placeholder(dtype=tf.float32,name='g_learning_rate')

    Gz = generator(gen_in)
    # _val = generator(_val_in,reuse=True)
    Dx = discriminator(real_in)
    Dg = discriminator(Gz, reuse=True)

    real_in_bgr = tf.map_fn(lambda img: RGB_TO_BGR(img), real_in)
    # real_in_bgr = tf.map_fn(lambda img: gray_to_rgb(img), real_in)
    # Gz_bgr = tf.map_fn(lambda img: gray_to_rgb(img), Gz)
    Gz_bgr = tf.map_fn(lambda img: RGB_TO_BGR(img), Gz)

    psnr=0
    ssim=0

    d_loss = 0.5*(tf.reduce_mean(Dx) + tf.reduce_mean(Dg))
    g_loss = tf.reduce_mean(Dg) +  STYLE_LOSS_FACTOR * get_style_loss(real_in_bgr, Gz_bgr)
    # g_loss = ADVERSARIAL_LOSS_FACTOR * -tf.reduce_mean(tf.log(Dg)) + PIXEL_LOSS_FACTOR * get_pixel_loss(real_in, Gz) \
    #          + STYLE_LOSS_FACTOR * get_style_loss(real_in_bgr, Gz_bgr) + SMOOTH_LOSS_FACTOR * get_smooth_loss(Gz)
    # g_loss = ADVERSARIAL_LOSS_FACTOR * -tf.reduce_mean(tf.log(Dg)) + PIXEL_LOSS_FACTOR * get_pixel_loss(real_in, Gz)
    # g_loss = get_pixel_loss(real_in, Gz)
    # g_loss = ADVERSARIAL_LOSS_FACTOR * -tf.reduce_mean(tf.log(Dg))

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    # d_solver = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars, global_step=global_step)
    d_solver = tf.train.AdamOptimizer(d_learning_rate).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_solver = tf.train.AdamOptimizer(g_learning_rate).minimize(g_loss, var_list=g_vars)
    # g_solver = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_vars)


    ##
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # add summary
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('ssim',ssim)
        # tf.summary.scalar('pix_loss', pix_loss)
        # tf.summary.scalar('smooth_loss', smooth_loss)
        # tf.summary.scalar('style_loss', style_loss)
        # tf.summary.image('input_noise',tf.reshape(gen_in,shape=[1,BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]]))
        # tf.summary.image('input_real',tf.reshape(real_in,shape=[1,BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]]))
        # tf.summary.image('output_Gz',tf.reshape(Gz,shape=[1,BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]]))
        writer = tf.summary.FileWriter(GRAPH_DIR, sess.graph)
        merged = tf.summary.merge_all()
        # end

        saver = initialize(sess)
        initial_step = global_step.eval()

        max_ssim = 0
        max_psnr = 0
        start_time = time.time()
        n_batches = 20
        total_iteration = n_batches * N_EPOCHS

        validation_batch = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), validation))
        # validation_batch = standardization(validation)        #标准化
        # validation_batch = standardization_1(validation)        #范围为(-1,1)
        getcycle(data_training,data_groundtruth)

        for index in range(initial_step, total_iteration):
            # input_batch = load_next_training_batch()
            # training_batch, groundtruth_batch = np.split(input_batch, 2, axis=2)

            training_batch = load_next_training_batch()
            groundtruth_batch = load_next_groundtruth_batch()

            # training_batch = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), training_batch))
            # groundtruth_batch = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), groundtruth_batch))
            training_batch = standardization(training_batch)          #标准化
            groundtruth_batch = standardization(groundtruth_batch)
            # training_batch = standardization_1(training_batch)       # 范围为(-1,1)
            # groundtruth_batch = standardization_1(groundtruth_batch)
            # _learning_rate = 0.95**(index/250)*0.05
            # _learning_rate_d = 0.95**(index/100)*0.02
            # _learning_rate_g = 0.95**(index/100)*0.02
            _learning_rate_d =  0.02
            _learning_rate_g =  0.02
            sess.run(d_clip)
            _, d_loss_cur = sess.run([d_solver, d_loss], feed_dict={gen_in: training_batch,real_in: groundtruth_batch,d_learning_rate:_learning_rate_d})

            _, g_loss_cur,summary_str = sess.run([g_solver, g_loss,merged], feed_dict={gen_in: training_batch,real_in: groundtruth_batch,g_learning_rate:_learning_rate_g})
            writer.add_summary(summary_str,index)

            # if  (index+1)%int(data_training.shape[0]/BATCH_SIZE)==0:
            #     np.random.seed(index)
            #     np.random.shuffle(data_training)
            #     np.random.seed(index)
            #     np.random.shuffle(data_groundtruth)
            #     getcycle(data_training,data_groundtruth)
            if math.isnan(d_loss_cur):
                return

            if (index + 1) % 200 == 0:
                saver.save(sess, CKPT_DIR, index)
            if (index + 1) % SKIP_STEP == 0:
                # saver.save(sess, CKPT_DIR, index)
                # image = sess.run(_val, feed_dict={_val_in: validation_batch})
                image = sess.run(Gz, feed_dict={gen_in: validation_batch})
                # image = np.resize(image[7][56:, :, :], [144, 256, 3])
                # image = np.reshape(image,[256,256,3])
                image = np.reshape(image,[256,256,3])
                ### average for image 3 channel
                image = average_image(image)
                image = np.reshape(image,[256,256])
                ###
                imsave('val_%d' % (index+1), image)
                image = skimage.io.imread(IMG_DIR+'val_%d.jpg' % (index+1)).astype('float32')
                psnr = measure.compare_psnr(metrics_image, image, data_range=255)
                ssim = measure.compare_ssim(metrics_image, image, multichannel=True, data_range=255, win_size=23)
                # if ssim>max_ssim and psnr>max_psnr:
                #     saver.save(sess, CKPT_DIR, index)
                #     max_ssim = ssim
                #     max_psnr = psnr
                print(
                    "Step {}/{} Gen Loss: ".format(index + 1, total_iteration) + str(g_loss_cur) + " Disc Loss: " + str(
                        d_loss_cur)+ " PSNR: "+str(psnr)+" SSIM: "+str(ssim))



if __name__=='__main__':
    data_training,data_groundtruth = training_dataset_init()
    validation = load_validation()
    print(validation.shape)
    train(data_training,data_groundtruth)
