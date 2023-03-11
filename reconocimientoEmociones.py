import cv2
import os
import numpy as np

def emotionImage(emotion):  #Metodo para reconocer la emocion y mandar la imagen correspondiente
    # Emojis
    if emotion == 'Felicidad': image = cv2.imread('Emojis/felicidad.jpeg')
    if emotion == 'Enojo': image = cv2.imread('Emojis/enojo.jpeg')
    if emotion == 'Sorpresa': image = cv2.imread('Emojis/sorpresa.jpeg')
    if emotion == 'Tristeza': image = cv2.imread('Emojis/tristeza.jpeg')
    return image

# ----------- Métodos usados para el entrenamiento y lectura del modelo ----------
#method = 'EigenFaces'
#method = 'FisherFaces'
method = 'LBPH'
#Lectura de los diferentes modelos previamente ya entrenados
if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

emotion_recognizer.read('modelo'+method+'.xml') #Leemos/probamos el modelo correspondiente
# --------------------------------------------------------------------------------
#Listamos las carpetas con las imagenes
dataPath = "C:/Users/Admin/OneDrive/Documentos/2022/Inteligencia_Artificial/ProyectoEmociones/Data"
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) #Stream del video

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')    #Detector de rostros

while True:
    #Leemos los fotrogramas y pasamos a escala de grises al momento de iniciar el streaming
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      #Escala de grises
    auxFrame = gray.copy()  #Copia limpia de cada fotograma despues de haber aplicado la escala de grises

    nFrame = cv2.hconcat([frame, np.zeros((480,300,3),dtype=np.uint8)]) #Los 2 frames, el de la camara y el de los emojis

    faces = faceClassif.detectMultiScale(gray,1.3,5)    #Almacenamos los rostros detectados

    for (x,y,w,h) in faces: #Recorremos los rostros identificados
        rostro = auxFrame[y:y+h,x:x+w]  #Coordenadas de cada rostro almacenado
        rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)    #Redimensionamos a 150x150
        result = emotion_recognizer.predict(rostro) #Comparamos el rostro con el modelo seleccionado anteriormente

        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)  #Mandamos un texto sobre el rostro detectado

        # EigenFaces
        if method == 'EigenFaces':
            if result[1] < 5700:    #Si el valor del resultado es menor se reconoce un rostro/emocion
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)    #Escribimos el texto sobre la imagen
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)       #Dibujamos un rectangulo alrededor del rostro
                image = emotionImage(imagePaths[result[0]]) #Mandamos la imagen reconocida que contiene la emocion
                nFrame = cv2.hconcat([frame,image]) #Imagen resultante del emoji
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        
        # FisherFace
        if method == 'FisherFaces':
            if result[1] < 500:
                #imagePath[result[0]] contiene el nombre de la emocion reconocida. El argumento para la funcion emotionImage
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])
        
        # LBPHFace
        if method == 'LBPH':
            if result[1] < 60:
                cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                image = emotionImage(imagePaths[result[0]])
                nFrame = cv2.hconcat([frame,image])
            else:
                cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                nFrame = cv2.hconcat([frame,np.zeros((480,300,3),dtype=np.uint8)])

    cv2.imshow('nFrame',nFrame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()