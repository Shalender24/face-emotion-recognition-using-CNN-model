from flask import Flask ,render_template ,request 
import cv2
import numpy as np 
from tensorflow.keras.models import model_from_json


json_file=open('emotiondetector.json','r')
model_json=json_file.read()
json_file.close()

model=model_from_json(model_json)
model.load_weights('emotiondetector.h5')


labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

app=Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload_image', methods=['GET','POST'])
def upload_image():
    if request.method != 'POST':
        return render_template('upload.html')
    else:
        uploaded_file=request.files["file"]
        if uploaded_file.filename != '':
            # image=cv2.imread('uploaded_file')

            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            gimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gimage=np.array(gimage)
            gimage=gimage.reshape(1,48,48,1)
            predict=model.predict(gimage)
            # print("gimage")
            return render_template('upload.html',output = labels[predict.argmax()])



if __name__=='__main__':
    app.run(debug=True)
