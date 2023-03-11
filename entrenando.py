import cv2
import os
import numpy as np
import time

def obtenerModelo(method,facesData,labels):
    #Metodos de reconocimiento que ofrece OpenCV
    if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
    if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
    if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Entrenando el reconocedor de rostros
    print("Entrenando ( "+method+" )...")
    inicio = time.time()    #Tiempo Inicial
    emotion_recognizer.train(facesData, np.array(labels))   #Entrenamos el reconocedor de rostros e indicamos el arreglo de los rostros y los labels de cada rostro
    tiempoEntrenamiento = time.time()-inicio                #Hacemos el calculo para saber el tiempo que se demora el entrenamiento
    print("Tiempo de entrenamiento ( "+method+" ): ", tiempoEntrenamiento)

    # Almacenando el modelo obtenido
    emotion_recognizer.write("modelo"+method+".xml")

dataPath = "C:/Users/Admin/OneDrive/Documentos/2022/Inteligencia_Artificial/ProyectoEmociones/Data"
emotionsList = os.listdir(dataPath)
print('Lista de emociones: ', emotionsList)
#Los diferentes arreglos para almacenar los rostros y sus etiquetas
labels = []
facesData = []
label = 0   #Las etiquetas para las 4 emociones, 0-3

for nameDir in emotionsList:
    emotionsPath = dataPath + '/' + nameDir

    for fileName in os.listdir(emotionsPath):               #Recorremos las imagenes contenidas en la carpeta de la emocion actual
        #print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)                                #Almacenamos el valor de la etiqueta en el otro arreglo labels
        facesData.append(cv2.imread(emotionsPath+'/'+fileName,0))   #Pasamos cada imagen a escala de grises (el valor cero)
        #image = cv2.imread(emotionsPath+'/'+fileName,0)
        #cv2.imshow('image',image)
        #cv2.waitKey(10)
    label = label + 1   #Incrementa en uno y pasamos a leer la siguiente carpeta

obtenerModelo('EigenFaces',facesData,labels)    #Llamamos al metodo para obtener los 3 modelos entrenados
obtenerModelo('FisherFaces',facesData,labels)
obtenerModelo('LBPH',facesData,labels)