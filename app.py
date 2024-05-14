from flask import Flask ,render_template ,request ,Response
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

def predict_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    try:
        for p, q, r, s in faces:
            face_image = gray[q : q + s, p : p + r]
            face_image = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image)
            img=np.array(img)
            img=img.reshape(1,48,48,1)
            img=img/255.0
            
            prediction_label = labels[pred.argmax()]
            cv2.rectangle(
                image, (p, q), (p + r, q + s), (255, 0, 0), 2
            )  # Draw rectangle around face
            cv2.putText(
                image,
                prediction_label,
                (p - 10, q - 10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 0, 255),
            )
        return image
    except cv2.error:
        return None

def generate_frames():
    camera = cv2.VideoCapture(
        0
    )  # Change the argument if using a different camera source

    while True:
        success, frame = camera.read()
        if not success:
            break

        processed_frame = predict_emotion(frame)  # Process the frame to predict emotion

        if processed_frame is not None:
            ret, buffer = cv2.imencode(".jpg", processed_frame)
            frame = buffer.tobytes()

            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    camera.release()


@app.route('/vedio-feed')
def vedio_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__=='__main__':
    app.run(debug=True)
