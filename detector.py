import cv2
import numpy as np
nombre_video_entrada = 'output_derecha.mp4'
nombre_video_salida = 'salida.mp4'
cap = cv2.VideoCapture(nombre_video_entrada)

if not cap.isOpened():
    prin(f"ERROR al abrir")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(nombre_video_salida, fourcc, fps, (width, height), isColor=True)

if not out.isOpened():
    print("no se puede guardar")
    cap.release()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    cuerdas_binario = cv2.adaptiveThreshold(
        src=blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=1
    )
    cuerdas_bgr = cv2.cvtColor(cuerdas_binario, cv2.COLOR_GRAY2BGR)
    #out.write(cuerdas_bgr)
    cv2.imshow('Cuerdas Adaptativas (Presiona q para salir)', cuerdas_binario)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
