from PIL import Image
from os import listdir

class_list = [
        'L2_10', 'L2_12', 'L2_15', 'L2_20', 'L2_21', 'L2_24', 'L2_25',
        'L2_27', 'L2_3', 'L2_30', 'L2_33', 'L2_34', 'L2_39', 'L2_40',
        'L2_41', 'L2_44', 'L2_45', 'L2_46', 'L2_50', 'L2_52']

for class_name in class_list:
    for img_filename in listdir("../dataset/"+class_name):
        with Image.open("../dataset/"+class_name+'/'+img_filename) as im:

            # Print all RGBA image filename, which are png file
            if im.mode == "RGBA":
                print(class_name+'/'+img_filename)
