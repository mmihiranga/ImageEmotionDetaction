from flask import Flask, render_template, Response,jsonify,request
from flask_cors import CORS, cross_origin
from keras.models import load_model
from time import sleep
import tensorflow as tf
from keras.preprocessing import image
import cv2
import numpy as np
import urllib.request

face_haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
model = load_model('./Emotion_Detection.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

camera = cv2.VideoCapture(0)
app=Flask(__name__)



def gen_frames(imagePath): 
            # Reading an image in default mode
            resp = urllib.request.urlopen(imagePath)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  
            print(len(faces_detected)>0)
            if(len(faces_detected)>0):

              for (x,y,w,h) in faces_detected:
                  
                  cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
                  roi_gray=gray_img[y:y+w,x:x+h]          #cropping region of interest i.e. face area from  image  
                  roi_gray=cv2.resize(roi_gray,(48,48))  
                  img_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)  
                  img_pixels = np.expand_dims(img_pixels, axis = 0)  
                  img_pixels /= 255  
          
                  predictions = model.predict(img_pixels)  
          
                  max_index = np.argmax(predictions[0])   #find max indexed array
          
                  emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  
                  predicted_emotion = emotions[max_index]  
                  cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
              return predicted_emotion  
            else:
              return "No Face Detected"
    


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/image_emotion', methods=['POST', 'GET'])
def image_emotion():
    imagePath = request.json['imagePath']
    if(imagePath == None):
        return jsonify({'Error': "Image Path Are Missing !"})
    else:
        pred = gen_frames(imagePath)
        if pred == None:
            return jsonify({'Error': "Error !"})
        else:
            return jsonify({'Prediction': gen_frames(imagePath)})


@app.route('/')
def index():
    return jsonify({'Hello': "Image!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)      


