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
| **5. Actualización Global**| `main.py` | El movimiento en píxeles se convierte a centímetros (`CM_POR_PX`) y se usa para actualizar la **posición global** . |
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


#### Parámetros Generales y de Video

| Parámetro             | Valor por defecto | Explicación                                                                                  |
|-----------------------|-------------------|----------------------------------------------------------------------------------------------|
| NOM_VID               | ""                | Nombre del archivo de video a procesar.                                                      |
| SKIP_RATE             | 1                 | Tasa de saltos de frames (1 = todos los frames, N = 1 de cada N).                            |
| START_FRAME           | 0                 | Frame de inicio para comenzar el procesamiento del video.                                   |
| N_FRAMES_HISTORIAL    | 5                 | Número de frames anteriores a mantener en el historial para ciertos cálculos (ej. seguimiento). |

#### Parámetros de Puntos y Seguimiento (Tracking no YOLO)

| Parámetro                 | Valor por defecto | Explicación                                                                                               |
|---------------------------|-------------------|-----------------------------------------------------------------------------------------------------------|
| RAD_PUN                   | 6                 | Radio de un punto o característica (visualización o área de influencia).                                  |
| UMB_DIST                  | 75                | Umbral de distancia para determinar si dos puntos son el mismo o si se ha movido significativamente.     |
| N_VEL_PR                  | 10                | Número de velocidades/vectores de trayectoria a promediar.                                                |
| MIN_SUPERVIVENCIA_FR      | 4                 | Mínimo de frames que un punto debe “sobrevivir” para ser considerado tracker válido.                      |
| FRAMES_MAX_ESTATICO       | 3                 | Máximo de frames consecutivos que un punto puede estar estático antes de ser descartado.                 |

#### Parámetros de Mapeo y Escala

| Parámetro                | Valor por defecto | Explicación                                                                                     |
|--------------------------|-------------------|-------------------------------------------------------------------------------------------------|
| Q_X                      | 6                 | Número de celdas en la dimensión X (cuadrícula).                                                |
| Q_Y                      | 5                 | Número de celdas en la dimensión Y (cuadrícula).                                                |
| SEP_CM                   | 2.5               | Separación real entre puntos/celdas (cm).                                                       |
| SEP_PX_EST               | 20                | Separación estimada en píxeles correspondiente a SEP_CM.                                        |
| CM_POR_PX                | 0.125             | Centímetros por píxel (SEP_CM / SEP_PX_EST). Coeficiente de escala.                             |
| FIXED_GRID_SIZE_CM       | 40.0              | Tamaño de la cuadrícula fija en centímetros.                                                   |
| RECT_SZ_CM_FALLBACK      | 30.0              | Tamaño de rectángulo de fallback en centímetros (cuando no hay información de tamaño).          |
| RECT_MARGIN_CM           | 5.0               | Margen adicional en centímetros para los rectángulos de detección.                             |
| MAP_PAD_PX               | 40                | Padding en píxeles alrededor del mapa para visualización.                                      |
| MAP_ESC_V                | 5.0               | Factor de escala para los vectores de velocidad en el mapa.                                     |
| MAP_CAN_SZ               | 800               | Tamaño del lienzo del mapa (ancho/alto) en píxeles.                                             |
| MAP_ZOOM_FACTOR          | 10                | Factor de zoom aplicado al mapa para visualización.                                             |

#### Parámetros de Visión Estéreo y Profundidad

| Parámetro                | Valor por defecto | Explicación                                                                                     |
|--------------------------|-------------------|-------------------------------------------------------------------------------------------------|
| PROFUNDIDAD_STEREO_ACTIVA| True              | Activar/desactivar cálculo de profundidad estéreo.                                              |
| BASELINE_CM              | 12.0              | Distancia entre centros ópticos de las cámaras estéreo (cm).                                   |
| FOCAL_PIX                | 800.0             | Distancia focal de la cámara en píxeles.                                                        |
| MIN_DEPTH_CM             | 20.0              | Profundidad mínima válida (cm).                                                                 |
| MAX_DEPTH_CM             | 300.0             | Profundidad máxima válida (cm).                                                                 |
| N_DEPTH_PR               | 5                 | Número de valores de profundidad a promediar.                                                   |
| MIN_DISPARITY            | 5                 | Disparidad mínima (píxeles) a considerar.                                                       |
| MAX_DISPARITY            | 150               | Disparidad máxima (píxeles) a considerar.                                                       |
| Y_TOLERANCE              | 6                 | Tolerancia vertical (píxeles) para emparejar puntos estéreo.                                    |

#### Parámetros de Detección por Color (Naranja)

| Parámetro             | Valor por defecto         | Explicación                                                                    |
|-----------------------|---------------------------|--------------------------------------------------------------------------------|
| ORANGE_HSV_LOW        | (5, 120, 150)             | Límite inferior HSV para color naranja.                                        |
| ORANGE_HSV_HIGH       | (22, 255, 255)            | Límite superior HSV para color naranja.                                        |
| ORANGE_MIN_AREA       | 30                        | Área mínima (px²) para contorno naranja válido.                                |
| ORANGE_MAX_AREA       | 5000                      | Área máxima (px²) para contorno naranja válido.                                |
| ORANGE_CIRCULARITY    | 0.4                       | Circularidad mínima requerida para objetos naranja.                            |

#### Parámetros de Procesamiento de Imágenes (Kernels)

| Parámetro             | Valor por defecto               | Explicación                                                                    |
|-----------------------|---------------------------------|--------------------------------------------------------------------------------|
| K_UNI_SIZE            | 5                               | Tamaño del kernel uniforme (5x5).                                              |
| K_LIMP_SIZE           | 3                               | Tamaño del kernel de limpieza (3x3).                                           |
| K_VERT_FILL_H         | 31                              | Altura del kernel vertical de relleno.                                         |
| K_VERT_FILL_W         | 3                               | Ancho del kernel vertical de relleno.                                          |
| K_UNI                 | np.ones((5, 5))                 | Kernel uniforme 5x5.                                                           |
| K_LIMP                | np.ones((3, 3))                 | Kernel de limpieza 3x3.                                                         |
| K_VERT_FILL           | cv2.getStructuringElement(...)  | Kernel rectangular vertical 3x31 para rellenar huecos.                         |
| MESH_CONSOLIDATE_K    | 7                               | Valor K para consolidación de malla.                                           |
| Y_MASK_OFFSET         | 100                             | Desplazamiento vertical para aplicación de máscara.                            |

#### Parámetros de Visualización y Colores

| Parámetro                  | Valor por defecto | Explicación                                                      |
|----------------------------|-------------------|------------------------------------------------------------------|
| ESC_VEC                    | 20                | Factor de escala para vectores de velocidad.                     |
| C_CAM                      | (0, 255, 255)     | Color de cámara (amarillo-cian BGR).                             |
| C_MAP_TXT                  | (255, 255, 255)   | Color del texto en el mapa (blanco).                             |
| C_MAP_ACT                  | (0, 0, 255)       | Color de objeto activo en mapa (rojo).                           |
| C_NARAN                    | (0, 165, 255)     | Color naranja (BGR).                                             |
| C_GRIS                     | (100, 100, 100)   | Color gris.                                                      |
| C_ACT                      | (0, 255, 0)       | Color de objeto activo (verde).                                  |
| C_DANO                     | (0, 0, 255)       | Color de objeto dañado (rojo).                                   |
| PORC_MOS_INT               | 100               | Porcentaje de información interna a mostrar.                     |
| PORC_MOS                   | 1.0               | Porcentaje de información a mostrar (decimal).                   |
| EDGE_POINT_RADIUS          | 2                 | Radio de puntos de borde.                                        |
| VISTA_MONO                 | False             | Habilitar vista monocular (ignorar estéreo).                     |
| MOSTRAR_VECTOR_SUPERVIVENCIA| True             | Mostrar vectores del tracker por supervivencia.                  |
| MOSTRAR_VECTOR_YOLO        | True              | Mostrar vectores del tracker YOLO.                               |

#### Parámetros de Detección de Daño

| Parámetro          | Valor por defecto | Explicación                                                             |
|--------------------|-------------------|-------------------------------------------------------------------------|
| DMG_ALPHA          | 0.1               | Factor de atenuación/peso en lógica de daño.                            |
| DMG_NUM_NB         | 4                 | Número de vecinos a considerar para evaluación de daño.                |
| DMG_FRAMES         | 6                 | Frames del historial usados para detección de daño.                     |
| DMG_THRESHOLD      | 2.5               | Umbral para clasificar como daño.                                       |
| DMG_DIST_TRACK     | 20                | Distancia de seguimiento usada en lógica de daño.                       |

#### Parámetros de Detecciónión y Seguimiento YOLO

| Parámetro             | Valor por defecto      | Explicación                                                                    |
|-----------------------|------------------------|--------------------------------------------------------------------------------|
| YOLO_MODEL_PATH       | "models/best.pt"       | Ruta del modelo YOLO entrenado.                                                |
| YOLO_TRACKING_ENABLED | True                   | Activar/desactivar tracking con YOLO.                                          |
| YOLO_SCALE_FACTOR     | 1.5                    | Factor de escala para bounding boxes de YOLO.                                  |
| YOLO_FRICTION         | 0.95                   | Coeficiente de fricción para suavizado de movimiento (YOLO).                  |
| YOLO_ACCELERATION     | 0.2                    | Coeficiente de aceleración para movimiento (YOLO).                             |

#### Parámetros de Exportación de Datos

| Parámetro                  | Valor por defecto            | Explicación                                                             |
|----------------------------|------------------------------|-------------------------------------------------------------------------|
| OUTPUT_JSON_YOLO           | "odometria_yolo.json"        | Archivo JSON con odometría calculada por YOLO.                          |
| OUTPUT_JSON_SUPERVIVENCIA  | "odometria_supervivencia.json"| Archivo JSON con odometría calculada por método de supervivencia.       |

#### Configuración de Actuadores Base

| Parámetro       | Valor por defecto                                      | Explicación                                      |
|-----------------|--------------------------------------------------------|--------------------------------------------------|
| Q_ACT_BASE      | [(1,1), (1,4), (2,1), (2,4), (3,1), (3,4)]              | Posiciones de los actuadores base en la cuadrícula. |

***


### 2. `utils.py` (Funciones necesarias)

Contiene funciones matemáticas y de ayuda usadas por los módulos de rastreo y dibujo.
#### Funciones Matemáticas Fundamentales

**Cálculo de Distancia Euclidiana**  
- **Función**: `dist(p1, p2)`  
- **Propósito**: Calcula la distancia euclidiana entre dos puntos 2D  
  $p_1 = (x_1, y_1)$ y $p_2 = (x_2, y_2)$  
  $$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

#### Funciones de Visualización y Fusión de Imágenes

**Conversión de Profundidad a Color**  
- **Función**: `depth_to_color(depth_cm, config)`  
- **Propósito**: Convierte un valor de profundidad (en cm) en un color BGR intuitivo.  
  - `MIN_DEPTH_CM` → Rojo intenso `(0, 0, 255)`  
  - `MAX_DEPTH_CM` → Azul intenso `(255, 0, 0)`  
  - Intermedios → Gradiente suave rojo a azul  
  - Fuera de rango → Rojo puro `(255, 0, 0)`

**Normalización de Vista de Celda**  
- **Función**: `normalize_cell_view(current_image, cell_target_size=100)`  
- **Propósito**: Redimensiona cualquier imagen a un tamaño fijo (por defecto 100×100 px) para visualización uniforme en rejillas. Usa `cv2.INTER_AREA` (óptima para reducción, resultados suaves).

**Fusión de Imágenes para Acumulación**  
- **Función**: `register_image_to_map(current_image, existing_image)`  
- **Propósito**: Combina una imagen nueva con un mapa acumulado existente mediante promedio ponderado 50%-50% (`cv2.addWeighted`). Usada para construir mapas persistentes a lo largo de múltiples frames.

#### Funciones de Mapeo y Transformación

**Cálculo de Transformación para Visualización de Mapa 2D**  
- **Función**: `map_trans(hist_m, m_w, m_h, config)`  
- **Propósito**: Determina escala y traslación para mostrar todos los puntos históricos (`hist_m`) centrados y visibles en un lienzo de tamaño `m_w × m_h`.  
  **Pasos**:  
  1. Bounding box de los puntos históricos  
  2. Factor de escala con padding  
  3. Límite superior mediante `MAP_ESC_V`  
  4. Offsets para centrado

#### Funciones de Entrada de Video Estéreo (ZED SVO)

**Apertura y Generador de Frames SVO**  
- **Función**: `open_svo_file(svo_path)`  
- **Propósito**: Abre un archivo `.svo` de cámara ZED y devuelve un generador de frames estéreo listos para procesar.  
  **Proceso**:  
  1. Inicializa ZED en modo reproducción  
  2. Recupera imagen izquierda + derecha por iteración  
  3. Concatena horizontalmente: `[Izquierda | Derecha]`  
  **Salida**: generador de frames, número total de frames y resolución combinada.  
  Es la función principal de carga de datos para todo el pipeline estéreo.

***

### 3. `stereo_processing.py` (Procesamiento Estéreo)

#### 1. Segmentación General de Objetos (`proc_seg`)
Procesa un frame para obtener una máscara binaria, eliminando ruido y unificando regiones cercanas.

**Pipeline principal**:
1. Conversión a escala de grises + suavizado Gaussiano (reducción de ruido).
2. Umbralización adaptativa gaussiana inversa → robusta a variaciones locales de iluminación.
3. Cierre morfológico con `k_uni` → une rupturas y rellena pequeños huecos.
4. Apertura morfológica con `k_limp` → elimina ruido.

> Soporta aceleración CUDA si `_cuda_processor` está disponible.

#### 2. Emparejamiento Estéreo de Contornos (`get_cns`)
Función crítica para obtener profundidad real a partir de visión estéreo.

**Proceso**:
- Detecta contornos en la máscara segmentada, priorizando agujeros internos y área ≥ 50 px.
- Filtra por ROI: solo centroides en cuadrantes activos (`config.Q_ACT_BASE`).
- Matching epipolar:
  - Tolerancia vertical: `|yL - yR| ≤ config.Y_TOLERANCE`
  - Disparidad: `x_izquierda - (x_derecha - ancho/2)`
  - Solo se aceptan disparidades en `[MIN_DISPARITY, MAX_DISPARITY]`

**Salida**: centroides izquierdos emparejados + sus disparidades válidas.

#### 3. Detección de Máscara de Malla/Red (`proc_mesh_mask`)
Genera máscara binaria específica para estructuras finas tipo malla o red (basada en textura).

**Pasos**:
- Trabaja sobre el canal S (saturación) del espacio HSV (más estable que V para texturas).
- Umbralización adaptativa en canal S.
- Consolidación con kernel cuadrado (`mesh_consolidate_k`).
- Cierre morfológico vertical con `k_vert_fill` (3×31) → conecta líneas verticales.
- Apertura final con `k_limp` para limpieza.

#### 4. Detección de Borde Superior de Malla (`get_mesh_boundary_y_pos`)
Determina la posición Y de la línea superior de la malla detectada.

**Método**:
- Obtiene bordes por sustracción: `mask - cv2.erode(mask)`
- Busca contornos grandes (área > 100) en la mitad superior de la imagen.
- Devuelve la mediana de las coordenadas Y mínimas → robusto frente a outliers.

#### 5. Detección de Marcadores Naranjas (`detect_orange_markers`)
Localiza balizas o marcadores circulares de color naranja.

**Pipeline**:
- Segmentación por rango HSV (`ORANGE_HSV_LOW` → `ORANGE_HSV_HIGH`)
- Apertura + cierre con kernel 3×3 para refinar máscara
- Filtrado geométrico:
  - Área entre `ORANGE_MIN_AREA` y `ORANGE_MAX_AREA`
  - Circularidad ≥ `ORANGE_CIRCULARITY`  
    $$ \text{circularidad} = \frac{4\pi \cdot \text{área}}{\text{perímetro}^2} $$

**Salida**: lista de diccionarios con:
{
    'cx': centro_x,
    'cy': centro_y,
    'area': área_px,
    'circularity': valor_circularidad,
    'bbox': (x, y, w, h)
}

***

### 4. `tracker.py` (Seguimiento de puntos)

**Clase principal**: `Tracker`  
Permite el seguimiento de puntos basado en el tiempo de superviciencia y la diferencia de posicion en cada frame. Sin dependencias externas (no usa SORT, DeepSORT ni ByteTrack).

#### Parámetros principales (configurables)
| Parámetro           | Origen en config                     | Uso                                           |
|---------------------|--------------------------------------|-----------------------------------------------|
| `max_d`             | `UMB_DIST`                           | Umbral máximo de distancia para asociación   |
| `len_v`             | `N_VEL_PR`                           | Longitud del historial de velocidades         |
| `FRAMES_MAX_ESTATICO`| —                                   | Frames consecutivos estático → eliminación    |
| `N_DEPTH_PR`        | —                                    | Ventana de historial para mediana de profundidad |

#### Flujo principal

1. **Predicción**  
   Para cada objeto existente:  
   `pos_pred = pos_actual + mediana(hist_vel)`  

2. **Matching greedy (más cercano primero)**  
   - Busca el contorno nuevo más cercano a la posición predicha  
   - Solo acepta si `dist < UMB_DIST`  
   - Cada contorno nuevo solo puede usarse una vez

3. **Actualización de objetos con match**
   - Actualiza historial de posiciones (`hist_pos`)
   - Detecta estado **estático** → si lleva `FRAMES_MAX_ESTATICO` sin moverse → supervivencia = 0
   - Si se mueve:
     - Calcula velocidad instantánea
     - Calcula profundidad:
       $$
       \text{depth}_\text{cm} = \frac{\text{FOCAL\_PIX} \times \text{BASELINE\_CM}}{\text{disparidad}}
       $$
     - Añade a `hist_depth` → valor final = **mediana** (muy estable)
     - Asigna color visual con `depth_to_color()` (rojo cerca, azul lejos)
     - Incrementa contador de supervivencia

4. **Gestión de objetos sin match**
   - Decrementa `supervivencia_fr`
   - Se mantienen solo si ≥ 1

5. **Creación de nuevos objetos**
   - Contornos no asociados → nuevo tracker con:
     - ID único
     - Supervivencia = 1
     - Historiales inicializados
     - Profundidad y color calculados desde el primer frame

6. **Limpieza final**
   - `self.objs` se actualiza eliminando todos con `supervivencia_fr < 1`

#### Estructura de cada objeto rastreado (dict)
{
    'id': int,
    'pos': (x, y),           # posición 2D izquierda
    'pos_R': (x_r, y_r),     # posición correspondiente derecha
    'depth_cm': float,       # profundidad suavizada (mediana)
    'color': (B, G, R),      # color según distancia
    'hist_vel': [(vx,vy)...],
    'hist_depth': [d...],
    'hist_pos': [(x,y)...],
    'supervivencia_fr': int
}

### 5. `drawing.py` (Visualización)

### 6. `main.py` (Bucle Principal)

### 7. `corrección` (Video de Entrada Recodificado)


Comando genérico (reemplace rutas y nombres de archivo según corresponda):

```
ffmpeg -y -i <archivo_entrada.mp4> -c:v libx264 -pix_fmt yuv420p -movflags +faststart <archivo_salida_h264.mp4>
```

Ejemplo aplicado al proyecto:

```
ffmpeg -y -i stereonr.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart stereo_h264.mp4
```

Esta conversión mantiene la calidad razonable y asegura compatibilidad con players web y contenedores que requieren `yuv420p` y `faststart`.
