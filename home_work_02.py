from functions import *
import warnings

warnings.filterwarnings('ignore')


# Define the directory of your images on Kaggle
dataset_dir = "img_align_celeba_25k"

# Get a list of all image paths in the directory
image_paths = glob.glob(os.path.join(dataset_dir, '*.jpg'))

print(image_paths)
# Considering only the first 20,000 images
image_paths = image_paths[:20000]

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

# Build and display discriminator summary
discriminator = build_discriminator()
discriminator.summary()

# Build and display generator summary
generator = build_generator(102)
generator.summary()

# Set noise dimension for generator input
noise_dimension = 102

# Build discriminator model
discriminator = build_discriminator()

# Build generator model
generator = build_generator(noise_dimension)

# Combine generator and discriminator to form the GAN model
gan_model = build_gan(generator, discriminator)

y_att_onehot = select_a_dominant_attribute("Male")

# Train the GAN model on the dataset and get the saved images list
[saved_images, G_model] = train(generator, discriminator, gan_model, dataset, noise_dimension, y_att_onehot,
                                num_epochs=250,
                                batch_size=256, display_frequency=1, verbose=0)
G_model.save('my_cDCGAN_CelebA_3.h5')

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
