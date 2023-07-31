import os
import shutil
import random




imagesPath1 = "Dataset/COVID"
imagesPath2 = "Dataset/NORMAL"

targetDir1 = "Dataset/Training/Covid"
targetDir2 = "Dataset/Training/Normal"

targetDir3 = "Dataset/Validation/Covid"
targetDir4 = "Dataset/Validation/Normal"

if not os.path.exists(targetDir1):
    os.mkdir(targetDir1)
    print("Training covid folder created")

if not os.path.exists(targetDir2):
    os.mkdir(targetDir2)
    print("Training normal folder created")

if not os.path.exists(targetDir3):
    os.mkdir(targetDir3)
    print("Validation covid folder created")

if not os.path.exists(targetDir4):
    os.mkdir(targetDir4)
    print("Validation normal folder created")



images_names = os.listdir(imagesPath1)
# print(images_names, "\n",len(images_names))
random.shuffle(images_names)

for i in range(int(len(os.listdir(imagesPath1)) * .8)):
    image_name = images_names[i]
    image_path = os.path.join(imagesPath1, image_name)

    target_path = os.path.join(targetDir1,image_name)

    shutil.copy2(image_path, target_path)
    print("Copying image: ", i)


for i in range(int(len(os.listdir(imagesPath1)) * .3)):
    image_name = images_names[i]
    image_path = os.path.join(imagesPath1, image_name)

    target_path = os.path.join(targetDir3,image_name)

    shutil.copy2(image_path, target_path)
    print("Copying image: ", i)



images_names = os.listdir(imagesPath2)
# print(images_names, "\n",len(images_names))
random.shuffle(images_names)

for i in range(int(len(os.listdir(imagesPath2)) * .8)):
    image_name = images_names[i]
    image_path = os.path.join(imagesPath2, image_name)

    target_path = os.path.join(targetDir2,image_name)

    shutil.copy2(image_path, target_path)
    print("Copying image: ", i)

for i in range(int(len(os.listdir(imagesPath2)) * .3)):
    image_name = images_names[i]
    image_path = os.path.join(imagesPath2, image_name)

    target_path = os.path.join(targetDir4,image_name)

    shutil.copy2(image_path, target_path)
    print("Copying image: ", i)


