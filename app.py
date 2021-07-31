import numpy as np
import os



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("edibility.h5")

@app.route('/')
def index():
    return render_template('index3.html')
    
    

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        preds = model.predict(x)
        
        print("prediction",preds)
            
        index = ['Asparagus. \n EDIBLE: Eating asparagus is good for hangovers!','Blue Vervain. \n EDIBLE: Vervain eases the feeling of anxiety and stress.','Cattail.\n EDIBLE: It get their name from the fuzzy, elongated seed heads that remind some of the tails of cats.','Chicory.\n EDIBLE: It is biennial plant (life span: 2 years) but it can survive up to 5 years under optimal conditions.','Fireweed.\n EDIBLE: Fireweed is often referred to as willowherb because its leaves resemble the willows.','green castor bean.\n POISONOUS: It is the very rich in chlorophyll and cytoplasm so produces high amount of oxygen']
        
        print(np.argmax(preds))
        
        text = "The plant in the uploaded image is  " + str(index[np.argmax(preds)])
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    