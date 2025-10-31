import cv2
import numpy as np

def procesarFrame(frame):
    """
    Procesa un frame de video:
    1. Convierte a escala de grises.
    2. Reduce ruido con filtro bilateral.
    3. Mejora contraste con CLAHE.
    4. Mejora nitidez con un kernel.
    """
    
    # 1. Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Reducción de ruido 
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 3. Mejorar contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)

    # 4. Mejorar nitidez
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    sharpened = cv2.filter2D(contrasted, -1, kernel)

    return sharpened

def calcularNitidez(frame):
    """
    Calcula la nitidez de un frame usando la varianza del Laplaciano.
    """
    
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        
    mean, stddev = cv2.meanStdDev(laplacian)
    
    return stddev[0][0] ** 2
