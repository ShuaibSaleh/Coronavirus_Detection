# from crypt import methods
from flask import Flask, flash, request, redirect, url_for, render_template
import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import urllib.request
import os
from werkzeug.utils import secure_filename

x_Ray_Image_detector = keras.models.load_model('CNN_X_Ray_Image_Detection.h5')
covid_19_detector = keras.models.load_model('CNN_Covid_19_Detection.h5')





app=Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'


app.secret_key = "cairocoders-ednalan"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/xraytest", methods=["POST","GET"])
def xraytest():
    if request.method == "POST" :
         
            file = request.files['myfile']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            path = '.' + url_for('static', filename='uploads/' + filename)

            img = image.load_img(path,target_size = (224,224))
            img = image.img_to_array(img)/255
            img = np.array([img])

            z = np.argmax(x_Ray_Image_detector.predict(img), axis=-1)

            if z == 0 :
                return render_template('NotXrayImage.html')
            else:
                z = np.argmax(covid_19_detector.predict(img), axis=-1)
                if z == 0:
                 return render_template('HasCovid.html')
                else:
                    return render_template('Good.html')
         
    else: 
       return render_template("xraytest.html")
    
            







if __name__=="__main__":
    app.run(debug=True ,port=5000) 







