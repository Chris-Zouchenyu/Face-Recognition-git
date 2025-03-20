import numpy as np
from PIL import Image
from .siamese import Siamese

class con():
    @staticmethod
    def contrast(path1,path2):
        model = Siamese()
        image_1 = path1
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')

        image_2 = path2
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
        probability = model.detect_image(image_1,image_2)
        return probability

# if __name__ == '__main__':
#     con.contrast(r'img\family1.jpg',r'img\nsh1.jpg')
# print("hello")
