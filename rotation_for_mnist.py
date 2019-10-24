#!/usr/bin/env python
# coding: utf-8

# In[3]:


from scipy import ndimage, misc
import numpy as np
import os
import cv2

def main():
    outPath = "D:\\rotated_class_data\\rotated_110"
    path = "C:\\Users\\Tan Ek Huat\\Desktop\\test_data\\3"

    # iterate through the names of contents of the folder
    for image_path in os.listdir(path):

        # create the full input path and read the file
        input_path = os.path.join(path, image_path)
        image_to_rotate = ndimage.imread(input_path)

        # reduce resolution
        f, e = os.path.splitext(path+item)
        imResize = im.resize((28,28))
        imResize.save(f + ' resized.jpg', 'JPG', quality=90)

        # create full output path, 'example.jpg' 
        # becomes 'rotate_example.jpg', save the file to disk
        fullpath = os.path.join(outPath, 'rotated_110'+image_path)
        misc.imsave(fullpath, rotated)

if __name__ == '__main__':
    main()


# In[ ]:




