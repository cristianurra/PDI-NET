import cv2
import numpy as np

nombre_video_entrada = 'output_derecha.mp4'
nombre_video_salida = 'salida_filtro_vibracion_sigma.mp4'

N_FRAMES_HISTORIAL = 50
UMBRAL_DESV_ESTANDAR = 7

CUADRANTES_X = 40
CUADRANTES_Y = 40
NUM_CUADRANTES = CUADRANTES_X * CUADRANTES_Y

cap = cv2.VideoCapture(nombre_video_entrada)

if not cap.isOpened():
    print(f"ERROR al abrir {nombre_video_entrada}")
    exit()

FRAME_INICIO = 200
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INICIO)
print(f"Iniciando la lectura del video en el frame: {FRAME_INICIO}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cuadrante_w = width // CUADRANTES_X
cuadrante_h = height // CUADRANTES_Y

historial_porcentajes = np.zeros((NUM_CUADRANTES, N_FRAMES_HISTORIAL), dtype=np.float32)

new_width = width
new_height = height

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(nombre_video_salida, fourcc, fps, (new_width, new_height), isColor=True)

if not out.isOpened():
    print("No se puede guardar el video de salida")
    cap.release()
    exit()

kernel_unir = np.ones((5, 5), np.uint8)
kernel_limpiar = np.ones((3, 3), np.uint8)

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
    cuerdas_cerradas = cv2.morphologyEx(cuerdas_binario, cv2.MORPH_CLOSE, kernel_unir)
    cuerdas_filtradas = cv2.morphologyEx(cuerdas_cerradas, cv2.MORPH_OPEN, kernel_limpiar)

    mascara_cuadrantes = np.zeros((height, width), dtype=np.uint8)

    idx_cuadrante = 0
    for i in range(CUADRANTES_Y):
        for j in range(CUADRANTES_X):
            y1 = i * cuadrante_h
            y2 = min((i + 1) * cuadrante_h, height)
            x1 = j * cuadrante_w
            x2 = min((j + 1) * cuadrante_w, width)

            cuadrante = cuerdas_filtradas[y1:y2, x1:x2]
            area_cuadrante = cuadrante.size

            if area_cuadrante > 0:
                blancos = np.sum(cuadrante == 255)
                porcentaje_actual = (blancos / area_cuadrante) * 100
            else:
                porcentaje_actual = 0.0

            historial_porcentajes[idx_cuadrante, :] = np.roll(historial_porcentajes[idx_cuadrante, :], shift=-1)
            historial_porcentajes[idx_cuadrante, -1] = porcentaje_actual

            desviacion_estandar = np.std(historial_porcentajes[idx_cuadrante, :])

            if desviacion_estandar < UMBRAL_DESV_ESTANDAR:
                mascara_cuadrantes[y1:y2, x1:x2] = 255

            idx_cuadrante += 1

    cuerdas_filtradas_final = cv2.bitwise_and(cuerdas_filtradas, mascara_cuadrantes)
    cuerdas_bgr = cv2.cvtColor(cuerdas_filtradas_final, cv2.COLOR_GRAY2BGR)

    out.write(cuerdas_bgr)
    cv2.imshow('Video con Filtro de Vibracion (Presiona q para salir)', cuerdas_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
