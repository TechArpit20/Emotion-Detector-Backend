import os
import time

from nltk.corpus import stopwords
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status, viewsets
from EmotionBackend.models import Person
from EmotionBackend.serializers import PersonSerializer, UserSerializer, ImageSerializer
from zipfile import ZipFile

from keras.models import load_model
from time import sleep
from keras.utils import img_to_array
from keras.preprocessing import image
import numpy as np
import cv2
import re
from PIL import Image
from nltk.stem.porter import PorterStemmer
import joblib
import sklearn
import warnings
warnings.filterwarnings('ignore')

class SignUpView(APIView):
    def get(self,request):
        users=Person.objects.all()
        response = PersonSerializer(users,many=True)
        return Response(response.data)

    def post(self, request):
        data= request.data
        try:
            user= Person.objects.get(username=data['username'])
            if user is not None:
                return Response({"error": "Username already exists"}, status=status.HTTP_208_ALREADY_REPORTED)
        except Person.DoesNotExist:
            try:
                user = Person.objects.get(email=data['email'])
                if user is not None:
                    return Response({"error": "Email already exists"}, status=status.HTTP_208_ALREADY_REPORTED)
            except Person.DoesNotExist:
                data['name']=data['name'].title().strip()
                data['username']= data['username'].lower().strip()
                data['email']= data['email'].lower().strip()
                serializer= UserSerializer(data=data)
                serializer.is_valid(raise_exception=True)
                serializer.save()
                return Response(serializer.data,status= status.HTTP_201_CREATED)


# Login API
class LoginView(APIView):
    def post(self, request):
        data= request.data
        try:
            user = Person.objects.get(email=data['email'].lower().strip())
        except Person.DoesNotExist:
            return Response({"error": "Invalid Email Id!!!"}, status=status.HTTP_406_NOT_ACCEPTABLE)
        user= UserSerializer(user).data
        if user.get('password')!= data['password']:
            return Response({'error':'Invalid Password!!!'}, status= status.HTTP_406_NOT_ACCEPTABLE)
        return Response(user,status=status.HTTP_200_OK)


# Image Processing
class DetectView(APIView):
    # serializer_class = ImageSerializer
    def post(self, request):
        face_classifier = cv2.CascadeClassifier('EmotionBackend/haarcascade_frontalface_default.xml')
        classifier = load_model('EmotionBackend/model.h5')
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        cap = cv2.VideoCapture(0)

        while True:
            _, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = classifier.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return Response({"message":label},status=status.HTTP_200_OK)

class TextView(APIView):
    def post(self, request):
        data= request.data
        lst = []
        my_model = joblib.load('EmotionBackend/lr_tfidf.sav')
        tfv = joblib.load('EmotionBackend/tfidf.sav')

        # sent = preprocessor(sentence)
        ps = PorterStemmer()

        review = re.sub('[^a-zA-Z]', ' ', data['sentence'])
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        sent= review

        lst.append(sent)
        sent_tfv = tfv.transform(lst)
        ans = my_model.predict(sent_tfv)
        mapped_ans = ''
        for i in ans:
            if i == 0:
                mapped_ans= 'Anger'
            elif i == 1:
                mapped_ans='Disgust'
            elif i == 2:
                mapped_ans='Fear'
            elif i == 3:
                mapped_ans='Joy'
            elif i == 4:
                mapped_ans='Neutral'
            elif i == 5:
                mapped_ans='Sadness'
            elif i == 6:
                mapped_ans='Surprise'

        return  Response({'message': mapped_ans}, status= status.HTTP_200_OK)