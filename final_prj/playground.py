import os
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt


# Join various path components
tumors_images_path = os.path.join(os.getcwd(), "data", "tumor", "resized")
notumor_images_path = os.path.join(os.getcwd(), "data", "notumor")
training_images_path = os.path.join(os.getcwd(), "data", "training", "images")
training_masks_path = os.path.join(os.getcwd(), "data", "training", "masks")


class ImageLoader:
    '''
    Takes path to image folder
    Loads images from folder and returns 4D numpy array (num_images, pixel_width, pixel_heights, channels)
    '''

    def __init__(self, image_path):
        self.image_path = image_path

    def load_image(self, path):
        ''' Loads single RGB image as np array and scale data to the range of [0, 1] '''
        color_image = Image.open(path)
        gray_image = np.array(ImageOps.grayscale(color_image), dtype=float) / 255
        return gray_image
        #color_image = np.array(Image.open(path), dtype=float) / 255
        #return color_image


    # load images, masks, and (if applicable) roi's
    def load_data(self, type, paths=None):
        ''' Loads all images '''

        # Create list of paths to each image file in the input path
        list_of_files = []
        for root, dirs, files in os.walk(self.image_path):
	        for file_ in files:
		        list_of_files.append(os.path.join(root, file_))

        # load each image in the list of file paths
        images = []
        for file_name in list_of_files:
            #color_image = self.load_image(file_name)
            #print(type(color_image))
            #gray_image = ImageOps.grayscale(color_image)
            #images.append(gray_image)
            images.append(self.load_image(file_name))


        # shapes each image to correct size
        if type == 'CNN':
            ''' loads 2d images '''
            images_all = images
            images_all_out = np.asarray(images_all)

            images_all_out = images_all_out.reshape(images_all_out.shape[0], 180, 180, 1)

        else:
            ''' loads 1d images '''
            images_all = []
            for i in range(len(list_of_files)):
                t = images[i].flatten()
                images_all.append(t)

            images_all_out = np.asarray(images_all)  # each element of the array is (1024,)

        if 'train' in self.image_path:
            print(f'Loaded {len(list_of_files)} images for the training set!')
        else:
            print(f'Loaded {len(list_of_files)} images for the validation set!')


        return images_all_out


image_loader_ = ImageLoader(tumors_images_path)
tumor_images = image_loader_.load_data('CNN')


new_img = tumor_images[1,:,:,0]
plt.imshow(new_img, cmap="gray")
#plt.gray()
plt.show()