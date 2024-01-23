import warnings

warnings.filterwarnings('ignore')

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import zeros, ones
from numpy.random import randn, randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers import BatchNormalization, Dropout, Reshape, Flatten

# Define the directory of your images on Kaggle
# dataset_dir = "/content/Kaggle/Celebrity_Faces_Dataset"
dataset_dir = "img_align_celeba_25k"

# Get a list of all image paths in the directory
image_paths = glob.glob(os.path.join(dataset_dir, '*.jpg'))

print(image_paths)
# Considering only the first 20,000 images
image_paths = image_paths[:20000]


# Create a function to open, crop and resize images
def load_and_preprocess_real_images(image_path, target_size=(64, 64)):
    # Open the image
    img = Image.open(image_path)
    # Crop 20 pixels from the top and bottom to make it square
    img = img.crop((0, 20, 178, 198))
    # Resize the image
    img = img.resize(target_size)
    # Convert to numpy array and scale to [-1, 1]
    img = np.array(img) / 127.5 - 1
    return img


# Open, crop and resize all images
dataset = np.array([load_and_preprocess_real_images(img_path) for img_path in image_paths])

# Print dataset shape
print(dataset.shape)

# Create a subplot for the first 25 images
fig, axes = plt.subplots(6, 6, figsize=(15, 16))

for i, ax in enumerate(axes.flat):
    # Get the i-th image
    img = dataset[i]
    # Rescale the image to [0, 1] for plotting
    img_rescaled = (img + 1) / 2
    # Plot the image on the i-th subplot
    ax.imshow(img_rescaled)
    ax.axis('off')

# Add a super title
fig.suptitle('Original Dataset Preprocessed Images', fontsize=25)

plt.tight_layout()
plt.show()

##

# Path to the attributes file : it has all pictures so initial 25000 are matched with image. 
attributes_file = "list_attr_celeba.txt"

# Initialize a dictionary to store the "Young" attribute for each image
young_attribute = {}
y_att = list()

# Read the file
with open(attributes_file, 'r') as file:
    # Read the number of images (first line)
    num_images = int(file.readline().strip())

    # Read the attributes names (second line) and find the index for "Young"
    attributes = file.readline().strip().split()
    young_index = attributes.index("Male")

    # Read each line, split it and extract the "Young" attribute
    for line in file:
        parts = line.strip().split()
        image_name = parts[0]
        young_value = int(parts[young_index + 1])  # +1 because of the image name
        young_attribute[image_name] = young_value
        y_att.append(young_value)

y_att_onehot = [[1, 0] if y == -1 else [0, 1] for y in y_att]


def build_discriminator(image_shape=(64, 64, 3)):
    model = Sequential()

    # Initial convolutional layer
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(LeakyReLU(0.2))

    # Second convolutional layer
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # Third convolutional layer
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # Fourth convolutional layer
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # Fifth convolutional layer
    model.add(Conv2D(512, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # Flatten and dense layer for classification
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    # Define optimizer and compile model
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


# Build and display discriminator summary
discriminator = build_discriminator()
discriminator.summary()


def build_generator(latent_dim, channels=3):
    model = Sequential()

    # Initial dense layer
    model.add(Dense(16 * 16 * 128, input_dim=latent_dim))
    model.add(LeakyReLU(0.2))

    # Reshape to (16, 16, 128) tensor for convolutional layers
    model.add(Reshape((16, 16, 128)))

    # First deconvolutional layer
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # Second deconvolutional layer
    model.add(Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(0.2))

    # Third deconvolutional layer
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # Fourth deconvolutional layer
    model.add(Conv2DTranspose(64, (4, 4), strides=(1, 1), padding='same'))
    model.add(LeakyReLU(0.2))

    # Output convolutional layer with 'tanh' activation
    model.add(Conv2D(channels, (8, 8), activation='tanh', padding='same'))

    return model


# Build and display generator summary
generator = build_generator(102)
generator.summary()


def build_gan(generator, discriminator):
    # Setting discriminator as non-trainable, so its weights won't update when training the GAN
    discriminator.trainable = False

    # Creating the GAN model
    model = Sequential()

    # Adding the generator
    model.add(generator)

    # Adding the discriminator
    model.add(discriminator)

    # Compiling the GAN model
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def generate_noise_samples(num_samples, noise_dim):
    X_noise = randn(noise_dim * num_samples)
    X_noise = X_noise.reshape(num_samples, noise_dim)
    return X_noise


def generate_images(epoch, generator, num_samples=6, noise_dim=102):
    """
    Generate images from the generator model for a given epoch.
    """
    # Generate noise samples
    X_noise = generate_noise_samples(num_samples, noise_dim)

    # Use generator to produce images from noise
    X = generator.predict(X_noise, verbose=0)

    # Rescale images to [0, 1] for visualization
    X = (X + 1) / 2

    return X


def display_saved_images(saved_images, display_frequency):
    """
    Display the saved generated images after training.
    """
    for epoch, images in enumerate(saved_images):
        fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
        for i, img in enumerate(images):
            axes[i].imshow(img)
            axes[i].axis('off')
        fig.suptitle(f"Generated Images at Epoch {epoch * display_frequency + 1}", fontsize=22)
        plt.tight_layout()
        plt.show()


def plot_generated_images(epoch, generator, num_samples=6, noise_dim=102, figsize=(15, 3)):
    """
    Plot and visualize generated images from the generator model for a given epoch.
    """

    # Generate noise samples
    X_noise = generate_noise_samples(num_samples, noise_dim)

    # Use generator to produce images from noise
    X = generator.predict(X_noise, verbose=0)

    # Rescale images to [0, 1] for visualization
    X = (X + 1) / 2

    # Plotting the images
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)

    for i in range(num_samples):
        axes[i].imshow(X[i])
        axes[i].axis('off')

    # Add a descriptive title
    fig.suptitle(f"Generated Images at Epoch {epoch + 1}", fontsize=22)
    plt.tight_layout()
    plt.show()

    save_dir = "saved_figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, f"generated_images_epoch_{epoch + 1}.png"))


def train(generator_model, discriminator_model, gan_model, dataset, noise_dimension,
          num_epochs=100, batch_size=128, display_frequency=10, verbose=1):
    # Create an empty list to store generated images for each epoch
    saved_images_for_epochs = []

    # Calculate the number of batches per epoch
    batches_per_epoch = int(dataset.shape[0] / batch_size)

    # Calculate half the size of a batch
    half_batch_size = int(batch_size / 2)

    # Loop over all epochs
    for epoch in range(num_epochs):
        # Loop over all batches within this epoch
        for batch_num in range(batches_per_epoch):

            # Generate a batch of real images and their corresponding labels

            ############### to find the index in batch
            sample_indices = randint(0, dataset.shape[0], half_batch_size)
            real_images = dataset[sample_indices]
            real_labels = ones((half_batch_size, 1))

            # Train the discriminator on the real images and calculate loss and accuracy
            dsr_loss_real, dsr_acc_real = discriminator_model.train_on_batch(real_images, real_labels)

            X_noise = randn(100 * half_batch_size)
            X_noise = X_noise.reshape(half_batch_size, 100)
            y_att = np.array(y_att_onehot)[sample_indices]
            X_noise = np.concatenate([X_noise, y_att], axis=1)

            fake_images = generator_model.predict(X_noise)
            fake_labels = zeros((half_batch_size, 1))

            # Train the discriminator on the fake images and calculate loss and accuracy
            dsr_loss_fake, dsr_acc_fake = discriminator_model.train_on_batch(fake_images, fake_labels)

            # Calculate the average discriminator loss and accuracy over real and fake images
            dsr_loss = 0.5 * np.add(dsr_loss_real, dsr_loss_fake)
            dsr_acc = 0.5 * np.add(dsr_acc_real, dsr_acc_fake)

            gan_noise = generate_noise_samples(batch_size, noise_dimension)
            gan_labels = np.ones((batch_size, 1))

            # Train the generator and calculate loss
            gen_loss, _ = gan_model.train_on_batch(gan_noise, gan_labels)

            if verbose:  # This condition checks if verbose is non-zero
                # Print training information for this batch
                print(
                    f"[ Epoch: {epoch + 1} , Batch: {batch_num + 1} ] --> [ Discriminator Loss : {dsr_loss:.6f} , Discriminator Accuracy: {100 * dsr_acc:.2f}% ] [ Generator Loss: {gen_loss:.6f} ]")

        # Display generated images at the specified frequency
        if epoch % display_frequency == 0:
            generated_images_for_epoch = generate_images(epoch, generator_model)
            saved_images_for_epochs.append(generated_images_for_epoch)

            # Plot generated images to visualize the progress of the generator
            plot_generated_images(epoch, generator_model)

    # Due to constraints on Kaggle output file size, saving the model is commented out.
    # generator_model.save('Photorealistic_Face_Generator.h5')

    return saved_images_for_epochs, generator_model


# Set noise dimension for generator input
noise_dimension = 102

# Build discriminator model
discriminator = build_discriminator()

# Build generator model
generator = build_generator(noise_dimension)

# Combine generator and discriminator to form the GAN model
gan_model = build_gan(generator, discriminator)

# Train the GAN model on the dataset and get the saved images list
[saved_images, G_model] = train(generator, discriminator, gan_model, dataset, noise_dimension, num_epochs=250,
                                batch_size=256, display_frequency=1, verbose=0)
G_model.save('cDCGAN_CelebA_3.h5')

'''
Feel free to change the model structure, adjust parameter and datasize accoroding to your preprerence (you can use original dataset). You can even take the code from any repository. 

Task 1

Select a dominant attribute (other than gender) from the 'list_attr_celeba.txt' file. Train a GAN model with this attribute and demonstrate control over the GAN outputs by manipulating the attribute.

Task 2 

Develop an enhanced strategy for a conditional GAN model that better differentiates between female and male images. 
Generate 20 images each for female and male by controlling the Z vector in your model. 
If the results aren't as expected, describe your approach and analyze potential reasons for the outcome. Then no deduction. 
I will give extra point for the exceptional outputs (up to 5 students).

'''

'''
import time

for i in range(0, 10):    
    z_noise = randn(100 * 1) / 3
    z_noise = z_noise.reshape(1, 100)
    z_con = np.concatenate([z_noise, np.array([[0, 1]])], axis=1)
    X = G_model.predict(z_con, verbose=0)
    X = (X + 1) / 2
    plt.imshow(np.squeeze(X))
    plt.show()
    print(i)
           
    
#[1, 0] = female, [0, 1] = male
z_noise = zeros(100).reshape(1, 100)
z_con = np.concatenate([z_noise, np.array([[1, 0]])], axis=1)
X = G_model.predict(z_con, verbose=0)
X = (X + 1) / 2
plt.imshow(np.squeeze(X))
plt.show()


model_1 = load_model("cDCGAN_CelebA_3.h5")
z_noise = randn(100 * 1) / 3
z_noise = z_noise.reshape(1, 100)
z_con = np.concatenate([z_noise, np.array([[1, 0]])], axis=1)
X = model_1.predict(z_con, verbose=0)
X = (X + 1) / 2
plt.imshow(np.squeeze(X))
plt.show()
'''
