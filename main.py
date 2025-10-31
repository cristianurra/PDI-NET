import cv2
import sys
import numpy as np

# Importar las funciones que acabamos de definir en processing.py
try:
    from processing import procesarFrame, calcularNitidez
except ImportError:
    print("Error: No se pudo encontrar el archivo 'processing.py'.", file=sys.stderr)
    print("Asegúrate de que 'processing.py' esté en el mismo directorio.", file=sys.stderr)
    sys.exit(1)


def main():
    # Comprobar argumentos de línea de comandos
    if len(sys.argv) < 2:
        print("Error: Debes proporcionar una fuente de video.", file=sys.stderr)
        print(f"Uso: python {sys.argv[0]} <fuente_de_video>", file=sys.stderr)
        sys.exit(1) 

    # Abrir el video
    video_source = sys.argv[1]
    vid = cv2.VideoCapture(video_source)


    if not vid.isOpened():
        print(f"Error abriendo la fuente de video: {video_source}", file=sys.stderr)
        sys.exit(1)

    # Crear ventanas
    cv2.namedWindow("Video izquierdo procesado", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Video derecho procesado", cv2.WINDOW_NORMAL)
   
    UMBRAL_NITIDEZ = 200

    while True:
        ret, img = vid.read()

        if not ret:
            print("Fin del video.")
            break

        # Dividir la imagen
        height, width = img.shape[:2]
        mid_x = width // 2 
        
        img_izq = img[:, 0:mid_x]
        img_der = img[:, mid_x:width]

        # Procesar ambas mitades
        proc_izq = procesarFrame(img_izq)
        nitidez_izq = calcularNitidez(proc_izq)

        proc_der = procesarFrame(img_der)
        nitidez_der = calcularNitidez(proc_der)

        
        if nitidez_izq < UMBRAL_NITIDEZ or nitidez_der < UMBRAL_NITIDEZ:
            continue # Saltar este frame
      
        print(f"Nitidez izq: {nitidez_izq}")
        print(f"Nitidez der: {nitidez_der}")
     
        cv2.imshow("Video izquierdo procesado", proc_izq)
        cv2.imshow("Video derecho procesado", proc_der)

        # Espera 10ms por una tecla. Si se presiona CUALQUIER tecla (!= -1), rompe el bucle.
        if cv2.waitKey(10) != -1:
            break
      
    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
