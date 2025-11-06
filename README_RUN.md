Ejecución del proyecto PDI-NET

Problema común:
- Si al ejecutar `python .\script\main.py` obtienes:
  ERROR: No se pudo abrir el video 'stereonr_h264.mp4'.

Causas y soluciones:
1) El archivo configurado en `script/config.py` (variable `NOM_VID`) no existe o no es legible por OpenCV
   - Opcion 1: Pasarle el archivo por comando. De la forma: 
      ``python .\script\main.py --video "C:\ruta\a\tu_video.mp4"``
   - Opción 2: Editar `script\config.py` y cambia `NOM_VID` por la ruta absoluta a un MP4 válido, por ejemplo:
     NOM_VID = r"C:\ruta\a\tu_video.mp4"


1) Tienes un archivo `.svo` (ZED) — OpenCV no lo abre directamente.
   Opciones:
   - Opción A (recomendada, gráfica): Abre el archivo `.svo` con ZED Explorer (parte del ZED SDK) y exporta a MP4 (File -> Export).
   - Opción B (programática): instala el ZED SDK y las bindings Python (pyzed), luego exporta a MP4 con un script. Ejemplo (requiere ZED SDK instalado):

     from pyzed import sl
     import cv2

     input_svo = r"C:\ruta\a\video.svo"
     out_mp4 = r"C:\ruta\a\salida.mp4"

     init = sl.InitParameters()
     init.set_from_svo_file(input_svo)
     cam = sl.Camera()
     status = cam.open(init)
     if status != sl.ERROR_CODE.SUCCESS:
         print('No se pudo abrir el SVO con ZED SDK')
     else:
         # Lee frames y escribe a un VideoWriter (simplificado, requiere más manejo real)
         # Alternativamente, usa ZED Explorer para exportar fácilmente.
         cam.close()

2) OpenCV no tiene soporte para el códec del MP4 en tu instalación.
   - Instala una build de OpenCV que incluya FFmpeg (por ejemplo `pip install opencv-python` suele funcionar) o reinstala OpenCV con soporte.

Prueba rápida (en PowerShell) para comprobar si OpenCV puede abrir un video:

```powershell
python - <<'PY'
import cv2
p = r"C:\ruta\a\tu_video.mp4"
cap = cv2.VideoCapture(p)
print('open:', cap.isOpened())
if cap.isOpened():
    print('FPS:', cap.get(cv2.CAP_PROP_FPS))
    print('W x H:', cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'x', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
PY
```

Pasos para correr el proyecto (Windows PowerShell):

```powershell
# Crear y activar virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Editar script\config.py para poner NOM_VID con la ruta absoluta a un MP4
# Ejecutar
python .\script\main.py
```

Si quieres, puedo:
- Sugerirte un comando exacto para exportar tu `.svo` si me das la ruta completa al archivo.
- Modificar `script/config.py` en el repositorio para apuntar a un MP4 que me indiques (puntero absoluto), o añadir soporte de línea de comandos para pasar la ruta del video al ejecutar `main.py`.
