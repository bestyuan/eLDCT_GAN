from model import *


def test(image,midname):
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    gen_in = tf.placeholder(shape=[None, 256, 256, BATCH_SHAPE[3]], dtype=tf.float32, name='generated_image')

    Gz = generator(gen_in)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver = initialize(sess)
        initial_step = global_step.eval()
        image = sess.run(tf.map_fn(lambda img: tf.image.per_image_standardization(img), image))
        image = sess.run(Gz, feed_dict={gen_in: image})
        image = average_image(image)
        image = np.reshape(image,[256,256])
        imsave('output'+midname, image)
        return image


if __name__ == '__main__':
    filelist=glob.glob('./dataset/test/*.jpg')
    for filename in filelist:
        midname = filename[filename.rindex('\\')+1:-4]
        image = np.array(skimage.io.imread(filename).astype('float32'))
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        image = np.expand_dims(image, axis=0)
        print(image[0].shape)
        test(image,midname)
