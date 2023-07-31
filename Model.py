from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tensorflow.keras.preprocessing import image




train_loc = './Dataset/Training/'
test_loc = './Dataset/Validation/'

# resize images 

trdata = ImageDataGenerator(rescale= 1./255,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True,
)

traindata = trdata.flow_from_directory(directory = train_loc, target_size = (224,224))

tsdata = ImageDataGenerator(rescale= 1./255,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True,
)

testdata = tsdata.flow_from_directory(directory = test_loc, target_size = (224,224))


print(traindata.class_indices)

# define input image
input_shape = (224,224,3)

# create the Network

# Input layer
img_imput = Input(shape  = input_shape, name = 'img_input')

# Convo layers
x = Conv2D(32, (3,3) , padding = 'same' , activation='relu', name = 'layer_1') (img_imput)
x = Conv2D(64, (3,3) , padding = 'same' , activation='relu', name = 'layer_2') (x)
x = MaxPool2D((2,2), strides=(2,2), name = 'layer_3') (x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3,3) , padding = 'same' , activation='relu', name = 'layer_4') (x)
x = MaxPool2D((2,2), strides=(2,2), name = 'layer_5') (x)
x = Dropout(0.25)(x)

x = Conv2D(128, (3,3) , padding = 'same' , activation='relu', name = 'layer_6') (x)
x = MaxPool2D((2,2), strides=(2,2), name = 'layer_7') (x)
x = Dropout(0.25)(x)


x = Flatten(name = 'fc_1')(x)
x= Dense(64, name = 'layer_8')(x)
x = Dropout(0.5) (x)
x = Dense(2, activation='sigmoid', name='predictions')(x)

# Generate the model
model = Model(inputs = img_imput, outputs =x , name='CNN_COVID_19')

# Print network structure
model.summary()


# Compiling the model
model.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy'])

# start Train/Test
batch_size = 32
hist = model.fit(traindata, steps_per_epoch = traindata.samples//batch_size,
                 validation_data = testdata,validation_steps = testdata.samples//batch_size,
                 epochs = 20
                 )




path = "./article--2020--04--20-0332--KRO_20-0332-02_ENG.jpg"

img = image.load_img(path, target_size = (224,224))
img = image.img_to_array(img)/255
img = np.array([img])


z= np.argmax(model.predict(img), axis=-1)
print(z)

#Save the model

model_filename = 'CNN_Covid_19_Detection.h5'
model.save(model_filename)



