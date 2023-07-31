import keras
from tensorflow.keras.preprocessing import image
import numpy as np


model1 = keras.models.load_model('CNN_X_Ray_Image_Detection.h5')
model2 = keras.models.load_model('CNN_Covid_19_Detection.h5')



path1 = "./article--2020--04--20-0332--KRO_20-0332-02_ENG.jpg"
#path2 = "./Dataset/NORMAL/normal_9.jpg"
path3 = "./download (25).jpg"

x = []

# for i in range(162):
    # path2 = "./Dataset/COVID/covid_"+str(i)+".jpg" 
img = image.load_img(path3, target_size = (224,224))
img = image.img_to_array(img)/255
img = np.array([img])


z= np.argmax(model1.predict(img), axis=-1)
print(model1.predict(img))
print(z)
if z == 0 :
        x.append("not xRay image")
else:
        z= np.argmax(model2.predict(img), axis=-1)
        if z == 0:
            x.append("covid")
        else:
            x.append("Normal")


print(len(x),x)