# CNN_MNIST_with_rotation
Building CNN model for handwritten digits with MNIST dataset and also with augmented dataset to observe how it affects the accuracy of the trained CNN model

This project is entirely done with tensorflow and python.

The procedure of the project was 

1. train the CNN model with MNIST handwritten digit dataset and obtain the accuracy.
2. augment the data by rotated all the digits and observe how it affects the accuracy.
3. augment the data by classes and seperate them to see the similarity of the error in each classe.
  e.g digits 0,1 and 8 will have higher accuracy when rotated 180 and -180 
4. reduce the resolution/image pixel from the original 28 all the way down to 2 by decresing 2 interval and observe the accuracy
