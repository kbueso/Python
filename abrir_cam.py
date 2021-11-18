import cv2 as cv
import functions
import os

cam = cv.VideoCapture(0) #Iniciando WebCam
file_name = "haarcascade_frontalface_alt2.xml"
classifier = cv.CascadeClassifier(f"{cv.haarcascades}{os.sep}{file_name}") #Modelo para reconhecer faces

dataframe = functions.load_dataframe() #Cargando dataframe com las imagenes para entrenamiento

X_train, y_train = functions.train_test(dataframe) #Dividindo conjuntos de treino e teste
pca = functions.pca_model(X_train) #Modelo PCA para extracion de contornos de imagen

X_train = pca.transform(X_train) #Conjunto de contornos extraídos

knn = functions.knn(X_train, y_train) #Entrenando con modelo de clasificacion KNN

#Rótulo de clasificaciones
label = {
    0: "Sin mascara",
    1: "Con mascara"
}

#Abriendo Webcam...
while True:
    status, frame = cam.read() #Leyendo imagen y extrayendo frame

    if not status:
        break

    if cv.waitKey(1) & 0xff == ord('q'):
        break
    
    #Transformando la imagen en escala de griz
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Detectando rostros en imagen
    faces = classifier.detectMultiScale(gray)

    #Iterando las caras encontradas
    for x,y,w,h in faces:
        gray_face = gray[y:y+h, x:x+w] #Recortando region de la cara

        if gray_face.shape[0] >= 200 and gray_face.shape[1] >= 200:
            gray_face = cv.resize(gray_face, (160,160)) #Redimensionando
            vector = pca.transform([gray_face.flatten()]) #Extrayendo contornos de la imagem

            pred = knn.predict(vector)[0] #Clasificando la imagen
            classification = label[pred]

            #Mostrando rectangulos alrededor del rostro
            if pred == 0:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
                print("\a")
            elif pred == 1:
                cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
            
            #Escribiendo clasificacion y cantidad de rostros vistos
            cv.putText(frame, classification, (x - 20,y + h + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)
            cv.putText(frame, f"{len(faces)} rostros identificados",(20,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)

    #Mostrando frame
    cv.imshow("Cam", frame)
