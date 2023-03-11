import cv2
import os
import imutils

#emotionName = 'Enojo'
#emotionName = 'Felicidad'
#emotionName = 'Sorpresa'
emotionName = 'Tristeza'

dataPath = "C:/Users/Admin/OneDrive/Documentos/2022/Inteligencia_Artificial/ProyectoEmociones/Data" 
emotionsPath = dataPath + '/' + emotionName

if not os.path.exists(emotionsPath):
    print('Carpeta creada: ',emotionsPath)
    os.makedirs(emotionsPath)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)     #Streaming del video

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')    #Detector de rostros con el algoritmo haarcascades
count = 0   #Contador para los rostros que se van guardando

while True:

    ret, frame = cap.read()
    if ret == False: break
    frame =  imutils.resize(frame, width=640)   #Redimensionamos los fotogramas a 640 pixeles
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Aplicamos escala de grises
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)    #Los rostros detectados

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)       #Dibujamos el cuadro que rodeara la cara
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)     #Redimensionamos los rostros a 150x150
        cv2.imwrite(emotionsPath + '/rotro_{}.jpg'.format(count),rostro)    #Almacenamos en la ruta definida para las carpetas y el formato de la imagen
        count = count + 1
    cv2.imshow('frame',frame)   #Mostramos el frame(recuadro del streaming)

    k =  cv2.waitKey(1)
    if k == 27 or count >= 200: #Definimos que tome 200 imagenes
        break

cap.release()
cv2.destroyAllWindows()