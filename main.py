from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import util
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

class ACGAN():
    def __init__(self, initial_epoch=0, gen_model_fn=None, dis_model_fn=None, latent_dim=100, use_colab=False, colab_path=None):
        self.initial_epoch=initial_epoch
        self.use_colab = use_colab
        self.colab_path = colab_path
        # Input shape
        self.img_rows = 60
        self.img_cols = 60
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 55
        self.latent_dim = latent_dim

        #noise for _sample_images
        np.random.seed(7)
        self.noise_for_sample_images = np.random.normal(0, 1, (10 * 10, self.latent_dim))

        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        if dis_model_fn != None: self.discriminator.load_weights(dis_model_fn)
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        if gen_model_fn != None: self.generator.load_weights(gen_model_fn)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)
        
        from matplotlib.pyplot import rcParams
        rcParams['figure.figsize'] = 14, 8

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 15 * 15, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((15, 15, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid", name='dense_valid')(features)
        label = Dense(self.num_classes, activation="softmax", name='dense_classes')(features)
        
        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        datloader = util.DatasetLoader()
        X_train, y_train = datloader.load_data()

        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # print(X_train.shape)
        # print(y_train.shape)
        # input()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        d_losses = []
        g_losses = []
        for epoch in range(self.initial_epoch, epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-9 
            img_labels = y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print('',end='\r')
            print ("%d [D loss: %f, fake_real_acc.: %.2f%%, pred_class_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0])
                , end='')
            d_losses.append(d_loss[0])
            g_losses.append(g_loss[0])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self._save_model()
                self._sample_images(epoch)
                self._plot_losses(epoch, d_losses, g_losses)


    def _sample_images(self, epoch):
        if epoch == 0:
            return

        plt.clf()

        r, c = 10 , 10
        noise = self.noise_for_sample_images
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        
        if self.use_colab:
            fig.savefig("%s/images/%d.png" % (self.colab_path,epoch))
        else:
            fig.savefig('images/%d.png' % (epoch,))
        plt.close()
    
    def sample_images_by_class(self, sample_per_class):
        for class_i,fdn in enumerate(os.listdir('sample_images_by_class')):
            noise = np.random.normal(0, 1, (r * c, self.latent_dim))

            labels = np.array([class_i] * 55)
            gen_imgs = self.generator.predict([noise, labels])
            gen_imgs = 1-(0.5 * gen_imgs + 0.5)
            for img_i in range(gen_imgs.shape[0]):
                result = Image.fromarray((gen_imgs[img_i,:,:,0] * 255).astype(np.uint8))
                result.save('sample_images_by_class/' + fdn + '/' + str(img_i) + '.bmp')

    def _plot_losses(self, epoch, d_losses, g_losses):
        plt.clf()
        plt.plot(d_losses, label='discriminator')
        plt.plot(g_losses, label='generator')
        plt.legend(loc='best')
        plt.title('epoch ' + str(epoch))

        if self.use_colab:
            plt.savefig('%s/losses.png' % (self.colab_path,))
        else:
            plt.savefig('losses.png')

    def _save_model(self):
        def save(model, model_name):
            for i in range(2):
                if i==0:
                    model_path = "saved_model/%s.json" % model_name
                    weights_path = "saved_model/%s.hdf5" % model_name
                elif i==1:
                    model_path = "saved_model/%s.json" % model_name
                    weights_path = "saved_model/%s.backup.hdf5" % model_name
                
                if self.use_colab:
                    model_path = self.colab_path + '/' + model_path
                    weights_path = self.colab_path + '/' + weights_path

                options = {"file_arch": model_path,
                            "file_weight": weights_path}
                json_string = model.to_json()
                open(options['file_arch'], 'w').write(json_string)
                model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")


if __name__ == '__main__':
    # acgan = ACGAN(initial_epoch=30250, dis_model_fn='saved_model/discriminator.hdf5', gen_model_fn='saved_model/generator.hdf5')
    acgan = ACGAN(latent_dim=300, dis_model_fn='saved_model/discriminator.hdf5', gen_model_fn='saved_model/generator.hdf5', initial_epoch=9400)
    acgan.train(epochs=99999, batch_size=100, sample_interval=50)
    # acgan.sample_images_by_class(40)