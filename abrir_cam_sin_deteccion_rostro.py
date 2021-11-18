import cv2 as cv
import functions
import os

cam = cv.VideoCapture(0) #Iniciando WebCam

dataframe = functions.load_dataframe() #Cargando dataframe com las imagenes para entrenamiento

X_train, y_train = functions.train_test(dataframe) #Dividiendo conjuntos para prueba
pca = functions.pca_model(X_train) #Modelo PCA para extracion de contornos de imagen

X_train = pca.transform(X_train) #Conjunto de contornos extraídos

knn = functions.knn(X_train, y_train) #Entrenando con modelo de clasificacion KNN

#Rótulo de clasificaciones
label = {
    0: "Sin mascara",
    1: "Con mascara"
}

#Abriendo webcam...
while True:
    status, frame = cam.read() #Leyendo imagen y extrayendo frame

    if not status:
        break

    if cv.waitKey(1) & 0xff == ord('q'):
        break
    
    #Transformando la imagen en escala de griz
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #Detectando rostros en imagen
    height, width, _ = frame.shape
    pt1, pt2 = ((width//2) - 100, (height//2) - 100), ((width//2) + 100, (height//2) + 100)
    region = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    gray_face = cv.cvtColor(region, cv.COLOR_BGR2GRAY)

    gray_face = cv.resize(gray_face, (160,160)) #Redimensionando
    vector = pca.transform([gray_face.flatten()]) #Extrayendo contornos de la imagem
    
    pred = knn.predict(vector)[0]
    classification = label[pred]

    color = (0,0,255)

    if pred == 1:
        color = (0,255,0)

    # Escribiendo clasificacion y cantidad de rostros vistos
    cv.putText(frame, classification, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv.LINE_AA)

    cv.rectangle(frame, pt1, pt2, color,thickness=3)
    #Mostrando frame
    cv.imshow("Cam", frame)
