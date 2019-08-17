import time

from model import *


def test(image,midname):
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    # gen_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='generated_image')
    gen_in = tf.placeholder(shape=[None, 256, 256, BATCH_SHAPE[3]], dtype=tf.float32, name='generated_image')
    # gen_in = tf.placeholder(shape=[None, 512, 512, BATCH_SHAPE[3]], dtype=tf.float32, name='generated_image')
    # real_in = tf.placeholder(shape=[None, BATCH_SHAPE[1], BATCH_SHAPE[2], BATCH_SHAPE[3]], dtype=tf.float32, name='groundtruth_image')
    real_in = tf.placeholder(shape=[None, 512, 512, BATCH_SHAPE[3]], dtype=tf.float32, name='groundtruth_image')

    Gz = generator(gen_in)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        saver = initialize(sess)
        initial_step = global_step.eval()

        start_time = time.time()
        n_batches = 200
        total_iteration = n_batches * N_EPOCHS

        image = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), image))
        # image = standardization(image)
        image = sess.run(Gz, feed_dict={gen_in: image})
        # image = np.resize(image[0][56:, :, :], [144, 256, 3])
        ## average for image 3 channel
        image = average_image(image)
        image = np.reshape(image,[256,256])
        ##

        # image = np.reshape(image,[256,256,3])
        imsave('output'+midname, image)
        return image

# def denoise(image):
#     image = scipy.misc.imread(image, mode='RGB').astype('float32')
#     npad = ((56, 56), (0, 0), (0, 0))
#     image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
#     image = np.expand_dims(image, axis=0)
#     print(image[0].shape)
#     output = test(image)
#     return output



if __name__ == '__main__':
    filelist=glob.glob('./dataset/test/*.jpg')
    for filename in filelist:
        midname = filename[filename.rindex('\\')+1:-4]
        # image = scipy.misc.imread(filename, mode='RGB').astype('float32')
        image = np.array(skimage.io.imread(filename).astype('float32'))
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # npad = ((56, 56), (0, 0), (0, 0))
        # image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
        image = np.expand_dims(image, axis=0)
        print(image[0].shape)
        test(image,midname)
