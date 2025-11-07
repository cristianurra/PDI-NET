# PDI-NET - Procesamiento de Imágenes Estéreo

## 🎥 Demo

[![Video Demo](https://img.youtube.com/vi/X3mn275QVqc/maxresdefault.jpg)](https://youtu.be/X3mn275QVqc)

---
![Interfaz](https://github.com/cristianurra/PDI-NET/blob/main/imagenes/captura.png)


## Ejecución en windows
```bash
python main.py -v "C:<ruta_al_video>" -sf 500
```
## Ejecución en Linux 
```bash
urra@linux:~$  python3 main.py -v "ruta_video.mp4" -sf 500 #iniciar en frame 500
```
## Videos de prueba
Github no permite subir videos (debido al tamaño de los archivos). Los videos stereo se encuentran en el siguiente [Enlace videos](https://drive.google.com/drive/folders/13DaTxRA60Ea-0Uf0BaMgEVquyJqHoPbr?usp=sharing)

## Cronología de Ejecución del Programa

***

### I. Inicialización (Paso Único)

El programa se prepara para comenzar el procesamiento.

1.  **Configuración:** **`main.py`** importa todas las constantes (`FOCAL_PIX`, `UMB_DIST`, etc.) al cargar los módulos **`config.py`** y **`utils.py`**.
2.  **Preparación:** Se inicializa la captura de video (`cv2.VideoCapture`).
3.  **Rastreador:** Se crea el objeto `tracker` (instancia de la clase `Tracker` de **`tracker.py`**).
4.  **Mapa:** La posición global $(x, y)$ se establece en el origen $(\mathbf{0.0}, \mathbf{0.0})$.
5.  **Interfaz:** Se crea la ventana principal de OpenCV.

***

### II. Bucle Principal por Frame (Ciclo Iterativo)

Este ciclo se repite para cada imagen del video (`while ret`):

| Paso | Módulo Clave | Descripción de la Acción |
| :--- | :--- | :--- |
| **1. Segmentación** | `stereo_processing.py` | La función `proc_seg` procesa el *frame* (gris, umbral, filtros) para generar una **imagen binaria** con los contornos de los objetos. |
| **2. Detección Estéreo** | `stereo_processing.py` | La función `get_cns` **empareja** los contornos entre el ojo izquierdo y el derecho, aplicando las restricciones de línea Y y de disparidad. |
| **3. Tracking y Profundidad**| `tracker.py` | El `Tracker` **predice** la posición, **asocia** los nuevos contornos, calcula la **profundidad** (`D = f*B/d`), y gestiona el historial y la supervivencia de los objetos. |
| **4. Odometría Visual**| `drawing.py` + `main.py`| La función `dib_mov` calcula el **vector de movimiento general de la cámara** (en píxeles) promediando las velocidades de los objetos rastreados. |
| **5. Actualización Global**| `main.py` | El movimiento en píxeles se convierte a centímetros (`CM_POR_PX`) y se usa para actualizar la **posición global** $(\text{pos\_m\_x}, \text{pos\_m\_y})$. |
| **6. Mapeo** | `main.py` + `utils.py` | La posición global determina la celda del *grid*. Se usan **`normalize_cell_view`** y **`register_image_to_map`** para actualizar la vista guardada en el historial de celdas. |
| **7. Dibujo (Overlay)** | `drawing.py` | Se dibujan sobre el *frame*: las líneas de ayuda, los puntos de objeto (coloreados por profundidad) y la flecha de movimiento de la cámara. |
| **8. Renderizado Final**| `drawing.py` | La función `dib_map` genera la imagen del mapa 2D. Finalmente, **`show_compuesta`** une los tres componentes (detección, segmentación y mapa) en una única interfaz. |

***

![Imagen binaria](https://github.com/cristianurra/PDI-NET/blob/main/imagenes/threshold.png)
![Imagen Stereo](https://github.com/cristianurra/PDI-NET/blob/main/imagenes/stereo.png)

### III. Cierre (Fin de Ejecución)

El programa termina cuando el usuario presiona la tecla **'q'** o se acaba el video. Se liberan los recursos (`cap.release()`) y se cierran las ventanas (`cv2.destroyAllWindows()`).




## Documentación del Código de Visión Estéreo y Mapeo 2D

Este proyecto implementa un sistema de visión por computadora que combina el **seguimiento de objetos**, el **cálculo de profundidad estéreo** y la **generación de un mapa 2D simple** a partir de un video que contiene imágenes estéreo (lado a lado).

***

### 1. `config.py` (Configuraciones y Constantes)

Este archivo contiene todos los parámetros numéricos y de configuración. 

| Constante | Descripción |
| :--- | :--- |
| `NOM_VID` | Nombre del archivo de video de entrada. |
| `UMB_DIST` | Umbral de distancia (en píxeles) para que el rastreador asocie un objeto existente con un nuevo contorno. |
| `FOCAL_PIX`, `BASELINE_CM` | Parámetros esenciales para la profundidad. `FOCAL_PIX` es la distancia focal; `BASELINE_CM` es la separación física entre las dos cámaras (en cm). |
| `MIN/MAX_DISPARITY` | Rango aceptable de disparidad (diferencia horizontal en píxeles) para un emparejamiento estéreo válido. |
| `FIXED_GRID_SIZE_CM` | El tamaño real (en cm) de cada celda en el mapa 2D de zonas visitadas. |
| `C_*` | Constantes BGR que definen los colores usados para dibujar la interfaz y los objetos. |

***

### 2. `utils.py` (Funciones de necesarias)

Contiene funciones matemáticas y de ayuda usadas por los módulos de rastreo y dibujo.

| Función | Propósito |
| :--- | :--- |
| `dist(p1, p2)` | Calcula la **distancia euclidiana** entre dos puntos en el plano. |
| `depth_to_color(depth_cm)` | Mapea un valor de profundidad (en cm) a un color BGR para visualización (rojo=cercano, azul=lejano). |
| `map_trans(hist_m, m_w, m_h)` | Calcula la **escala** y el **desplazamiento** necesarios para centrar y encajar el área explorada del mapa dentro del lienzo. |
| `normalize_cell_view(...)` | Redimensiona una porción de la imagen a un tamaño estándar para guardarla como vista de una celda del mapa. |
| `register_image_to_map(...)` | Combina una nueva vista de una celda con su imagen ya existente usando un promedio ponderado. |

***

### 3. `stereo_processing.py` (Procesamiento Estéreo)

Maneja la segmentación inicial de la imagen y la lógica fundamental de emparejamiento.

| Función | Descripción |
| :--- | :--- |
| `proc_seg(frame, k_uni, k_limp)` | Realiza la **segmentación** de objetos mediante umbralización adaptativa y operaciones morfológicas (cierre y apertura) para limpiar contornos. |
| `get_cns(cns_filt, q_w, q_h, w)` | 1. **Contornos:** Encuentra contornos válidos dentro de las Regiones de Interés (ROI). 2. **Emparejamiento Estéreo:** Intenta emparejar contornos del ojo izquierdo con los del ojo derecho, aplicando la **restricción de línea de barrido** (tolerancia Y) y la **restricción de disparidad**. 3. Devuelve los centroides emparejados y su disparidad. |

***

### 4. `tracker.py` (Seguimiento de puntos)

Contiene la clase `Tracker`, que gestiona la persistencia de los objetos, su movimiento y sus propiedades.

| Clase / Método | Propósito y Lógica |
| :--- | :--- |
| `class Tracker` | Mantiene el estado de cada objeto rastreado (historial de posición, velocidad y profundidad). |
| `update_and_get(...)` | **Algoritmo de seguimiento:** Predice la posición, asocia nuevos contornos usando `UMB_DIST`, actualiza la posición, velocidad, profundidad (**$D = \frac{f \cdot B}{d}$**), y gestiona la supervivencia del objeto (detección de estático). |

***

### 5. `drawing.py` (Visualización)

Agrupa todas las funciones responsables de dibujar la información y la interfaz de usuario.

| Función | Elementos que dibuja |
| :--- | :--- |
| `dib_escala_profundidad(...)` | La barra de color lateral que muestra la escala de profundidad. |
| `dib_mov(...)` | Dibuja los puntos rastreados y estima el **vector de movimiento general de la cámara** (Odometría Visual) a partir de los objetos. Muestra la profundidad promedio y el movimiento en texto. |
| `dib_ayu(...)` | Dibuja líneas de referencia, etiquetas de "Ojo Izq/Der" y los límites de las Regiones de Interés (ROI). |
| `dib_map(...)` | Renderiza el **mapa 2D**. Dibuja las celdas visitadas con la imagen guardada y muestra la posición actual y el área de visión de la cámara. |
| `show_compuesta(...)` | Ensambla las diferentes partes (`frame` principal, segmentación, y mapa) y las muestra en una única ventana. |

***

### 6. `main.py` (Bucle Principal)

Define el flujo de ejecución por *frame* del programa.

1. **Captura y Procesamiento:** Lee el *frame*, llama a `proc_seg`, `get_cns`, y `tracker.update_and_get`.
2. **Cálculo de Odometría:** Usa el vector de movimiento de la cámara estimado y lo traduce a centímetros usando `CM_POR_PX` para actualizar la posición global $(\text{pos\_m\_x}, \text{pos\_m\_y})$.
3. **Actualización del Mapa:** La posición global se usa para determinar la celda $(\text{grid\_x}, \text{grid\_y})$ y se actualiza el diccionario `hist_celdas_vis` con la vista actual.
4. **Visualización:** Llama a todas las funciones de `drawing.py` para construir y mostrar la interfaz.

### 7. `corrección` (Video de Entrada Recodificado)

Se reencodificó el video de entrada a H.264 para mejorar compatibilidad y reproducción en distintos reproductores/servicios. El nombre del fichero usado en el flujo pasó de `stereonr.mp4` a `stereo_h264.mp4`.

Comando genérico (reemplace rutas y nombres de archivo según corresponda):

```
ffmpeg -y -i <archivo_entrada.mp4> -c:v libx264 -pix_fmt yuv420p -movflags +faststart <archivo_salida_h264.mp4>
```

Ejemplo aplicado al proyecto:

```
ffmpeg -y -i stereonr.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart stereo_h264.mp4
```

Esta conversión mantiene la calidad razonable y asegura compatibilidad con players web y contenedores que requieren `yuv420p` y `faststart`.
