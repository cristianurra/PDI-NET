# 📚 Wiki - Sistema de Procesamiento Estéreo PDI-NET

## 📋 Índice
1. [Arquitectura General](#arquitectura-general)
2. [Módulos del Sistema](#módulos-del-sistema)
3. [Flujo de Datos](#flujo-de-datos)
4. [Diagrama de Interacción](#diagrama-de-interacción)

---

## 🏗️ Arquitectura General

El sistema PDI-NET es una aplicación de procesamiento de video estéreo en tiempo real que combina múltiples técnicas de visión por computadora:

- **Visión Estéreo**: Procesamiento de pares de imágenes para calcular profundidad
- **Tracking de Objetos**: Seguimiento de contornos mediante supervivencia
- **Detección YOLO**: Identificación de bordes y nudos en redes de pesca
- **Odometría Visual**: Estimación de movimiento de cámara
- **Detección de Anomalías**: Identificación de daños en redes
- **Mapeo 2D/3D**: Visualización de trayectorias y posición global

### Tecnologías Principales:
- **Python 3.9+**
- **OpenCV**: Procesamiento de imágenes
- **PyTorch + CUDA**: Inferencia YOLO
- **Tkinter**: Interfaz gráfica
- **Open3D**: Visualización 3D
- **NumPy**: Operaciones numéricas
- **ZED SDK**: Manejo de archivos SVO

---

## 📦 Módulos del Sistema

### 1. **main.py** - Punto de Entrada
**Propósito**: Inicializa la aplicación y lanza la interfaz gráfica.

**Responsabilidades**:
- Cargar configuración global
- Inicializar ventana Tkinter
- Pasar control a `gui.py`

**Interacciones**:
- ➡️ `config.py`: Lee configuración
- ➡️ `gui.py`: Lanza interfaz
- ➡️ `hardware_optimizer.py`: Detecta capacidades CUDA

**Código Clave**:
```python
config = ConfiguracionGlobal()
root = tk.Tk()
app = StereoAppTkinter(root, config)
root.mainloop()
```

---

### 2. **config.py** - Configuración Global
**Propósito**: Centralizar todos los parámetros del sistema.

**Parámetros Principales**:

#### Procesamiento de Video:
- `NOM_VID`: Ruta del video a procesar
- `START_FRAME`: Frame inicial
- `SKIP_RATE`: Procesar 1 de cada N frames
- `VISTA_MONO`: True=vista simple, False=estéreo

#### Visión Estéreo:
- `FOCAL_PIX`: Distancia focal (píxeles)
- `BASELINE_CM`: Separación entre cámaras (cm)
- `MIN_DISPARITY`, `MAX_DISPARITY`: Rango de disparidad
- `Y_TOLERANCE`: Tolerancia vertical para matching

#### Tracking:
- `UMB_DIST`: Umbral de distancia para asociar objetos
- `MIN_SUPERVIVENCIA_FR`: Frames mínimos de supervivencia
- `CM_POR_PX`: Conversión píxeles → centímetros (0.125)

#### YOLO:
- `YOLO_MODEL_PATH`: Ruta al modelo .pt
- `YOLO_TRACKING_ENABLED`: Activar/desactivar YOLO
- `YOLO_CONF_THRESHOLD`: Confianza mínima (0.83)
- `YOLO_ACCELERATION`: Suavizado de velocidad (0.3)
- `YOLO_FRICTION`: Decaimiento de inercia (0.85)

#### Detección de Daños:
- `DMG_ALPHA`: Factor adaptativo de umbral
- `DMG_THRESHOLD`: Multiplicador de área vecina (1.5)
- `DMG_FRAMES`: Frames para confirmar daño (3)

#### Visualización:
- `C_OBJ`: Color de objetos trackeados (verde)
- `C_DANO`: Color de daños (rojo)
- `MOSTRAR_VECTOR_SUPERVIVENCIA`: Mostrar vector de movimiento
- `MOSTRAR_VECTOR_YOLO`: Mostrar vector YOLO

**Interacciones**:
- ⬅️ Todos los módulos leen esta configuración
- ✏️ `gui.py` modifica valores en tiempo real

---

### 3. **gui.py** - Interfaz Gráfica y Orquestador
**Propósito**: Interfaz Tkinter y thread principal de procesamiento.

**Componentes**:

#### **Clase `ProcesadorEstereoThread`**:
Thread que ejecuta el pipeline de procesamiento.

**Atributos**:
```python
self.config: ConfiguracionGlobal
self.mapeo: GlobalMapper2D          # Mapa 2D de posición
self.tracker: Tracker               # Tracking de supervivencia
self.damage_detector: DamageDetector # Detección de daños
self.yolo_tracker: YOLOTracker      # Tracking YOLO
self.visual_odometry: VisualOdometry # Odometría YOLO
self.odometry_drawer: AdaptiveTrajectoryDrawer # Gráfico 2D

# Datos de tracking
self.matrices_yolo: List            # Matrices 4x4 para Open3D
self.matrices_supervivencia: List
self.trajectory_supervivencia: List # Trayectoria 2D
self.yolo_markers: List            # Marcadores de bordes/nudos
self.damage_log: List              # Log de daños detectados
```

**Pipeline de Procesamiento** (método `run()`):
```
1. Abrir video (MP4/SVO)
2. Para cada frame:
   ├─ Segmentación (proc_seg)
   ├─ Detección de malla (proc_mesh_mask)
   ├─ Extracción de contornos (get_cns)
   ├─ Tracking de supervivencia (tracker.update)
   ├─ YOLO tracking (yolo_tracker.track_frame)
   │  ├─ Actualizar odometría (visual_odometry.update)
   │  ├─ Guardar matriz YOLO
   │  └─ Detectar marcadores (cruce de tercio central)
   ├─ Detección de daños (damage_detector.detect)
   ├─ Actualizar mapeo (mapeo.update_position)
   ├─ Dibujar visualizaciones
   │  ├─ Vectores de movimiento (dib_mov)
   │  ├─ Vector YOLO (dib_vector_yolo)
   │  ├─ Mapa 2D (dib_map)
   │  ├─ Radar 3D (mapeo.draw_map)
   │  └─ Gráfico odometría (odometry_drawer.draw)
   └─ Actualizar GUI (actualizar_gui)
```

#### **Clase `StereoAppTkinter`**:
Interfaz gráfica principal.

**Layout**:
```
┌─────────────────────────────────────────────────┐
│ Video Estéreo (75%)     │ Controles (25%)       │
│ ┌─────────────────────┐ │ ┌─────────────────┐   │
│ │                     │ │ │ Máscara Binaria │   │
│ │   Frame Principal   │ │ │                 │   │
│ │                     │ │ └─────────────────┘   │
│ └─────────────────────┘ │ ┌─────────────────┐   │
│ ┌──────────┬──────────┐ │ │ Mapa de Zonas  │   │
│ │  Radar   │ Odometría│ │ │      2D        │   │
│ │   3D     │  Visual  │ │ └─────────────────┘   │
│ └──────────┴──────────┘ │ ┌─────────────────┐   │
│ [Pausar][▶][Mapa 3D]   │ │  Parámetros     │   │
│ Frame: 1234 / 5000      │ │  [sliders...]   │   │
│ ━━━━━━━━━━━━━━━━━ 25%   │ └─────────────────┘   │
└─────────────────────────────────────────────────┘
```

**Funcionalidades**:
- `start_processing_thread()`: Inicia procesamiento
- `pause_thread()` / `resume_thread()`: Control de reproducción
- `show_3d_map()`: Visualizador Open3D
- `change_video()`: Cambiar video sin cerrar app
- `guardar_reporte()`: Exportar CSV + imagen de daños

**Interacciones**:
- ➡️ `stereo_processing.py`: Procesamiento de imágenes
- ➡️ `tracker.py`: Tracking de supervivencia
- ➡️ `mapper.py`: Mapeo 2D
- ➡️ `yolo_tracker.py`: Detección YOLO
- ➡️ `visual_odometry.py`: Odometría
- ➡️ `anomaly_detector.py`: Detección de daños
- ➡️ `drawing.py`: Visualizaciones

---

### 4. **stereo_processing.py** - Procesamiento de Imágenes
**Propósito**: Algoritmos de visión estéreo y segmentación.

**Funciones Principales**:

#### `proc_seg(frame, K_UNI, K_LIMP)`
Segmentación de red de pesca usando Canny + morfología.
```python
Input: frame BGR, kernels morfológicos
Output: Máscara binaria con contornos
Pipeline:
  1. Convertir a escala de grises
  2. Canny edge detection (100, 200)
  3. Unión morfológica (K_UNI)
  4. Limpieza morfológica (K_LIMP)
```

#### `proc_mesh_mask(frame, consolidate_k, K_LIMP, K_VERT_FILL)`
Detecta la red como región sólida.
```python
Input: frame BGR + kernels
Output: Máscara de la red (blanco=red, negro=fondo)
Pipeline:
  1. Threshold adaptativo en canal L (LAB)
  2. Componente conectada más grande
  3. Cierre morfológico para llenar huecos
```

#### `get_mesh_boundary_y_pos(mesh_mask, x, max_y, K_LIMP)`
Encuentra el borde superior de la red en columna X.

#### `get_cns(cns_filt, q_w, q_h, w, config, y_max_track)`
**Función crítica**: Extrae contornos y hace matching estéreo.
```python
Input: Máscara de contornos, parámetros
Output: 
  - Contornos izquierdos con match
  - Pares (contorno_L, contorno_R)
  - Contornos con disparidad
  
Algoritmo:
  1. Dividir frame en izquierdo/derecho
  2. Submuestrear según PORC_MOS
  3. Encontrar contornos en cada lado
  4. Matching por disparidad:
     - Misma fila Y (±Y_TOLERANCE)
     - Disparidad en rango [MIN, MAX]
     - Contorno derecho a la izquierda del izquierdo
  5. Calcular profundidad: depth = (focal * baseline) / disparity
```

#### `detect_orange_markers(frame)`
Detecta marcadores naranjas (HSV filtering).

**Optimización CUDA**:
Si hay GPU disponible, usa procesamiento paralelo mediante `cuda_processor`.

**Interacciones**:
- ⬅️ `config.py`: Parámetros de procesamiento
- ⬅️ `hardware_optimizer.py`: Procesamiento CUDA
- ➡️ `gui.py`: Retorna contornos procesados

---

### 5. **tracker.py** - Tracking de Supervivencia
**Propósito**: Seguimiento temporal de objetos detectados.

**Clase `Tracker`**:

**Concepto**: Asocia contornos entre frames usando proximidad espacial.

**Atributos**:
```python
self.tracked_objects: List[Dict]  # Objetos activos
self.next_id: int                 # ID secuencial
self.n_vel_pr: int                # Frames para promedio de velocidad
```

**Estructura de Objeto**:
```python
{
    'id': int,                    # ID único
    'pos': (x, y),               # Posición actual
    'hist_pos': [(x,y), ...],    # Historial de posiciones
    'hist_vel': [(vx,vy), ...],  # Historial de velocidades
    'depth_cm': float,           # Profundidad
    'supervivencia': int         # Frames sin actualizar
}
```

**Método `update_and_get(matched_pairs)`**:
```python
Entrada: Lista de pares (contorno_L, contorno_R)

Para cada par:
  1. Calcular centroide y profundidad
  2. Buscar objeto cercano (distancia < UMB_DIST)
  3. Si existe match:
     - Actualizar posición
     - Calcular velocidad
     - Resetear supervivencia
  4. Si no hay match:
     - Crear nuevo objeto con ID único

Para objetos sin match:
  - Incrementar supervivencia
  - Si supervivencia > MIN_SUPERVIVENCIA_FR: eliminar

Retorno: Lista de objetos actualizados
```

**Estrategia de Supervivencia**:
- Permite que objetos persistan temporalmente sin detección
- Útil cuando hay oclusiones o fallos de detección momentáneos
- Balance: `MIN_SUPERVIVENCIA_FR` (típicamente 5-10 frames)

**Interacciones**:
- ⬅️ `stereo_processing.py`: Recibe contornos matched
- ➡️ `mapper.py`: Pasa objetos para mapeo
- ➡️ `gui.py`: Retorna objetos para visualización

---

### 6. **mapper.py** - Mapeo Global 2D
**Propósito**: Estimar posición y orientación global de la cámara.

**Clase `GlobalMapper2D`**:

**Concepto**: Usar movimiento de múltiples objetos para estimar odometría de cámara.

**Atributos**:
```python
self.global_x: float = 0.0        # Posición X (cm)
self.global_y: float = 0.0        # Posición Y (cm)
self.global_angle: float = 0.0    # Orientación (radianes)
```

**Método `update_position(tracked_objects)`**:
```python
Algoritmo de Odometría Visual:

1. Recolectar puntos con historial:
   - Necesita ≥2 posiciones previas
   - prev_pts = posición en frame t-1
   - curr_pts = posición en frame t

2. Estimar transformación Afín (si ≥3 puntos):
   M = estimateAffinePartial2D(curr → prev)
   │
   ├─ Traslación: dx, dy = -M[0,2], -M[1,2]
   ├─ Rotación: dθ = atan2(M[1,0], M[0,0])
   └─ Escala: s = sqrt(M[0,0]² + M[1,0]²)

3. Fallback centroide (si <3 puntos):
   dx = mean(prev_x) - mean(curr_x)
   dy = mean(prev_y) - mean(curr_y)

4. Filtro de ruido:
   Si distancia² > 2500 → descartar (teletransportación)

5. Transformar a coordenadas globales:
   dx_global = dx * cos(θ) - dy * sin(θ)
   dy_global = dx * sin(θ) + dy * cos(θ)
   
   Escalar: dx_global *= CM_POR_PX
           dy_global *= CM_POR_PX

6. Actualizar pose global:
   global_x += dx_global
   global_y += dy_global
   global_angle += dθ * 0.5  # Factor de suavizado
```

**Método `draw_map(objs, frames_history)`**:
Genera visualización tipo radar con:
- Objetos actuales (círculos verdes)
- Trayectorias históricas (líneas)
- Vectores de velocidad (flechas)
- Grid de referencia
- Rango de visión

**Interacciones**:
- ⬅️ `tracker.py`: Recibe objetos trackeados
- ⬅️ `config.py`: Parámetros de mapeo
- ➡️ `gui.py`: Retorna imagen de radar

---

### 7. **yolo_tracker.py** - Detección YOLO
**Propósito**: Detectar y trackear bordes/nudos usando YOLOv11.

**Clase `YOLOTracker`**:

**Modelo**:
- YOLOv11 entrenado custom
- 2 clases: `0=Borde`, `1=Nudo`
- Tracker: `botsort.yaml` (ByteTrack + OSNet)

**Método `track_frame(frame)`**:
```python
Entrada: Frame BGR (vista izquierda)

Pipeline:
  1. Inferencia YOLO:
     results = model.track(frame, persist=True, classes=[0,1])
  
  2. Para cada detección:
     box = (x1, y1, x2, y2)
     cx, cy = centro del bounding box
     track_id = ID persistente del tracker
     class_id = 0 o 1
  
  3. Detección de tercio central:
     tercio_izq = ancho * 1/3
     tercio_der = ancho * 2/3
     
     Si tercio_izq ≤ cx ≤ tercio_der:
       Y (no estaba antes O primera detección):
         ➡️ Marcar como "crossed_center"
  
  4. Calcular vectores de movimiento:
     Si existe posición previa:
       vector_x = cx_actual - cx_previo
       vector_y = cy_actual - cy_previo
  
  5. Anotar frame con bounding boxes

Salida:
  - frame_anotado
  - vectors_x: [vx1, vx2, ...]
  - vectors_y: [vy1, vy2, ...]
  - detections: [{class, name, cx, cy, crossed_center, id}, ...]
```

**Sistema de Marcadores**:
- Detecta cuando objeto ENTRA al tercio central (33%-66%)
- No importa altura Y (arriba, centro, abajo)
- `prev_x_positions` mantiene estado entre frames
- Evita duplicados: solo marca al entrar, no en cada frame

**Clase `YOLOOverlayDrawer`**:
Dibuja overlay de detecciones en frame.

**Interacciones**:
- ⬅️ `config.py`: Ruta del modelo, thresholds
- ➡️ `visual_odometry.py`: Pasa vectores de movimiento
- ➡️ `gui.py`: Retorna detecciones y marcadores

---

### 8. **visual_odometry.py** - Odometría Visual YOLO
**Propósito**: Estimar movimiento de cámara usando tracking YOLO.

**Clase `VisualOdometry`**:

**Concepto**: Usar movimiento de objetos detectados por YOLO para inferir movimiento de cámara.

**Física del Modelo**:
```
Objetos estáticos en mundo → si se mueven en imagen, cámara se movió

Si objeto se mueve DERECHA en imagen:
  → Cámara se movió IZQUIERDA

Velocidad cámara = -mediana(velocidades_objetos)
```

**Método `update(vectors_x, vectors_y)`**:
```python
Entrada: Vectores de movimiento de objetos YOLO

Algoritmo de Física:
  1. Calcular movimiento promedio:
     avg_dx = median(vectors_x)
     avg_dy = median(vectors_y)
  
  2. Invertir (objeto → cámara):
     target_vel_x = -avg_dx * CM_POR_PX
     target_vel_y = -avg_dy * CM_POR_PX
  
  3. Aplicar aceleración suave (filtro):
     vel_x = vel_x * (1-α) + target_vel_x * α
     vel_y = vel_y * (1-α) + target_vel_y * α
     donde α = YOLO_ACCELERATION (0.3)
  
  4. Si no hay detecciones:
     Fricción: vel *= YOLO_FRICTION (0.85)
     Si |vel| < 0.05: vel = 0
  
  5. Actualizar posición:
     pos_x += vel_x
     pos_y += vel_y
  
  6. Guardar en trayectoria:
     trajectory.append((pos_x, pos_y))
```

**Estados**:
- `TRACKING ACTIVO` (verde): Hay detecciones YOLO
- `INERCIA` (cian): Sin detecciones, usando fricción
- `INICIALIZANDO` (gris): Estado inicial

**Clase `AdaptiveTrajectoryDrawer`**:
Dibuja gráfico 2D de trayectorias.

**Características**:
- Zoom automático a rango de datos
- Dos trayectorias simultáneas:
  - Verde: YOLO
  - Azul: Supervivencia
- Marcadores de bordes/nudos:
  - Amarillo: Borde (clase 0)
  - Magenta: Nudo (clase 1)
- Vector de velocidad actual (flecha roja)
- Info de posición y rango

**Método `draw(...)`**:
```python
Pipeline de Renderizado:
  1. Calcular límites (min/max X, Y)
  2. Agregar padding (10%)
  3. Calcular escala: px/cm
  4. Función world_to_screen(wx, wy):
     sx = (wx - min_x) / range_x * drawable_width
     sy = (wy - min_y) / range_y * drawable_height
  
  5. Dibujar trayectorias:
     - Verde con gradiente (más reciente = más brillante)
     - Azul para supervivencia
  
  6. Dibujar marcadores:
     - Obtener posición desde trajectory[frame_index]
     - Círculo coloreado + número de marcador
  
  7. Dibujar posición actual (círculo + outline)
  8. Dibujar vector velocidad (flecha)
  9. Leyendas y texto de estado
```

**Interacciones**:
- ⬅️ `yolo_tracker.py`: Recibe vectores de movimiento
- ➡️ `gui.py`: Proporciona trayectoria y gráfico

---

### 9. **anomaly_detector.py** - Detección de Daños
**Propósito**: Identificar agujeros/daños en la red de pesca.

**Clase `DamageDetector`**:

**Algoritmo de Detección**:
```python
Pipeline:
  1. Threshold Adaptativo (blockSize=15):
     - Convertir a LAB
     - Adaptive threshold en canal L
     - Limpieza morfológica (3x3)
  
  2. Identificar Red Principal:
     - Componentes conectadas
     - Seleccionar área más grande = red
     - Cierre morfológico (11x11)
  
  3. Detectar Agujeros:
     holes_mask = NOT(red_mask)
     - Componentes conectadas en agujeros
     - Filtrar los que tocan bordes
  
  4. Análisis Estadístico:
     Para cada agujero i:
       área_i, centroide_i, bbox_i
     
     mean_área = promedio(áreas)
     std_área = desviación(áreas)
     
     z_factor = 3 * (1 - exp(-ALPHA * N_agujeros))
     threshold = mean_área + z_factor * std_área
  
  5. Criterio de Daño:
     Para agujeros con área > threshold:
       - Encontrar K vecinos más cercanos
       - max_vecino = max(áreas_vecinos)
       - Si área > DMG_THRESHOLD * max_vecino:
           ➡️ Candidato a daño
  
  6. Tracking Temporal:
     Asociar candidatos entre frames (distancia < DMG_DIST_TRACK)
     Incrementar contador si persiste
     Si contador ≥ DMG_FRAMES (3):
       ➡️ Confirmar daño
       ➡️ Asignar ID único
```

**Sistema de Supervivencia de Daños**:
```python
Estado Candidato:
  [centroide, bbox, frame_count, id, área, max_vecino_área]

Transición:
  frame_count = 0     → Candidato nuevo
  frame_count = 1-2   → En evaluación
  frame_count ≥ 3     → Daño confirmado (asignar ID)
```

**Visualización**:
- Rectángulo rojo con transparencia
- Label: "DMG #ID"
- Solo daños confirmados

**Método `_draw_damage(img, candidate)`**:
```python
overlay = imagen.copy()
cv2.rectangle(overlay, bbox, COLOR_ROJO, -1)  # Relleno
cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)  # Alpha blend
cv2.rectangle(img, bbox, COLOR_ROJO, 2)  # Borde
cv2.putText(img, f"DMG #{id}", ...)
```

**Interacciones**:
- ⬅️ `config.py`: Parámetros de detección
- ➡️ `gui.py`: Retorna frame con daños + metadata

---

### 10. **drawing.py** - Funciones de Visualización
**Propósito**: Dibujar overlays y gráficos en frames.

**Funciones Principales**:

#### `dib_ayu(frame, w, h, q_w, q_h, config)`
Dibuja líneas auxiliares de referencia.
```python
- Línea vertical central (verde)
- Grid de submuestreo (azul tenue)
- Margen de profundidad mínima (rojo)
```

#### `dib_mov(frame, objs, w, h, depth_cm, config, show_vector)`
**Función crítica**: Dibuja objetos trackeados y vectores de movimiento.
```python
Para cada objeto:
  1. Dibujar círculo en posición actual
     Color según supervivencia:
       - Verde brillante: recién detectado
       - Verde oscuro: en supervivencia
  
  2. Si show_vector:
     - Calcular velocidad promedio (hist_vel)
     - Escalar: vel * SCALE_FACTOR
     - Dibujar flecha desde objeto
  
  3. Label con ID y profundidad
  
  4. Calcular movimiento de cámara:
     - Promedio de velocidades de todos objetos
     - Proyectar: vector global de movimiento
  
Retorno: (del_x, del_y, vista_limpia)
```

#### `dib_vector_yolo(frame, w, h, vx, vy, config)`
Dibuja vector de movimiento YOLO.
```python
Posición: (w/4, h-50)  # Esquina inferior izquierda
Escala: vel * SCALE_FACTOR
Color: Amarillo/Cian según modo
Arrow: TipLength = 0.3
```

#### `dib_escala_profundidad(frame, w, h, config)`
Barra de escala de profundidad (colormap jet).

#### `dib_map(hist_celdas, pos_x, pos_y, grid_sz, ...)`
Genera mapa 2D de zonas visitadas.
```python
Canvas negro + grid

Para cada celda visitada:
  1. Obtener imagen registrada
  2. Calcular posición en mapa
  3. Dibujar imagen miniatura
  4. Borde según profundidad (colormap)

Dibujar posición actual (círculo rojo)
Dibujar trayectoria reciente (línea verde)
```

**Interacciones**:
- ⬅️ Todos los módulos de procesamiento
- ➡️ `gui.py`: Retorna frames anotados

---

### 11. **hardware_optimizer.py** - Optimización de Hardware
**Propósito**: Detectar y aprovechar aceleración por hardware.

**Clase `HardwareOptimizer`**:

**Detección de Capacidades**:
```python
1. CUDA (NVIDIA GPU):
   - PyTorch: torch.cuda.is_available()
   - OpenCV: cv2.cuda.getCudaEnabledDeviceCount()
   - Info: Nombre GPU, memoria, compute capability

2. CPU:
   - Núcleos físicos vs lógicos
   - Threads disponibles para OpenCV

3. YOLO:
   - device='cuda:0' si disponible
   - device='cpu' como fallback
```

**Clase `CudaProcessor`**:
Procesamiento paralelo de imágenes en GPU.

**Métodos**:
- `process_stereo_pair()`: Procesamiento estéreo acelerado
- `batch_canny()`: Edge detection en batch
- `morphological_ops()`: Operaciones morfológicas GPU

**Función `initialize_hardware_optimization()`**:
```python
Retorna: (HardwareOptimizer, CudaProcessor|None)

Si CUDA disponible:
  - Crear CudaProcessor
  - Inyectar en stereo_processing
  - Configurar YOLO para GPU
Sino:
  - Configurar para CPU
  - Optimizar threads OpenCV
```

**Interacciones**:
- ➡️ `yolo_tracker.py`: Device para modelo
- ➡️ `stereo_processing.py`: Procesador CUDA
- ⬅️ `gui.py`: Inicializa optimizaciones

---

### 12. **utils.py** - Utilidades Generales
**Propósito**: Funciones auxiliares compartidas.

**Funciones**:

#### `open_svo_file(path)`
Abre archivos SVO (ZED SDK).
```python
Retorna: (generador_frames, total_frames, width, height)

Yield por cada frame:
  - Lee frame del SVO
  - Convierte a formato OpenCV
  - Maneja errores de lectura
```

#### `normalize_cell_view(img, cell_target_size)`
Normaliza imagen de celda para mapa.
```python
1. Resize a tamaño fijo
2. Ecualización de histograma (CLAHE)
3. Recortar bordes
Retorna: Imagen normalizada
```

#### `register_image_to_map(new_img, existing_img)`
Registra imagen nueva con existente.
```python
Si existe imagen previa:
  1. Feature matching (ORB)
  2. Encontrar homografía
  3. Warp nueva imagen a referencia
  4. Blend (alpha=0.7)
Sino:
  Retornar nueva imagen
```

**Interacciones**:
- ➡️ Todos los módulos que necesitan funciones comunes

---

## 🔄 Flujo de Datos

### Flujo Principal de Procesamiento

```
                    ┌─────────────┐
                    │   main.py   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  config.py  │◄──────────┐
                    └──────┬──────┘           │
                           │                  │
                    ┌──────▼──────┐          │
                    │   gui.py    │          │
                    │ (Tkinter)   │          │
                    └──────┬──────┘          │
                           │                  │
                    ┌──────▼──────┐          │
                    │ Processing  │          │
                    │   Thread    │          │
                    └──────┬──────┘          │
                           │                  │
          ┌────────────────┼────────────────┐│
          │                │                ││
   ┌──────▼──────┐  ┌──────▼──────┐  ┌─────▼▼────┐
   │  stereo_    │  │  yolo_      │  │ anomaly_  │
   │ processing  │  │  tracker    │  │ detector  │
   └──────┬──────┘  └──────┬──────┘  └─────┬─────┘
          │                │                │
   ┌──────▼──────┐  ┌──────▼──────┐       │
   │   tracker   │  │   visual_   │       │
   │  (survival) │  │  odometry   │       │
   └──────┬──────┘  └──────┬──────┘       │
          │                │                │
          │         ┌──────▼──────┐        │
          └────────►│   drawing   │◄───────┘
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  GUI Update │
                    │  (Tkinter)  │
                    └─────────────┘
```

### Flujo de Datos de Tracking

```
Frame Estéreo
    │
    ├──► proc_seg() ──────────► Máscara binaria
    │                                │
    ├──► proc_mesh_mask() ─────────► Máscara de red
    │                                │
    └──► get_cns() ─────────────────┴─► Contornos con disparidad
              │                              │
              │                              ▼
              │                         Calcular profundidad
              │                              │
              └──────────────────────────────┤
                                             │
                    ┌────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │  Tracker.update()   │
         │  - Asociar objetos  │
         │  - Calcular vel     │
         │  - Supervivencia    │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────────┐
         │ GlobalMapper2D.update() │
         │ - Estimar pose global   │
         │ - Transformación afín   │
         └──────────┬──────────────┘
                    │
                    ▼
            Posición global (X, Y, θ)
```

### Flujo de YOLO + Odometría

```
Frame Izquierdo
    │
    ├──► yolo_tracker.track_frame()
    │         │
    │         ├──► Inferencia YOLOv11
    │         ├──► Tracking (BoTSORT)
    │         ├──► Calcular vectores movimiento
    │         └──► Detectar tercio central
    │                   │
    │         ┌─────────┴─────────┐
    │         │                   │
    │    vectors_x,y         detections
    │         │                   │
    │         ▼                   ▼
    │    VisualOdometry       Marcadores
    │         │                   │
    │         ├──► update()       │
    │         │    - Física       │
    │         │    - Aceleración  │
    │         │    - Fricción     │
    │         │                   │
    │         ▼                   │
    │    Trayectoria             │
    │    (pos_x, pos_y)          │
    │         │                   │
    │         └─────────┬─────────┘
    │                   │
    └──► anomaly_detector.detect()
              │
              ▼
         Daños detectados
              │
              ▼
    ┌─────────────────────┐
    │ AdaptiveTrajector   │
    │ yDrawer.draw()      │
    │ - Graficar verde    │
    │ - Graficar azul     │
    │ - Dibujar marcadores│
    └─────────────────────┘
```

---

## 📊 Diagrama de Interacción Completo

```
┌───────────────────────────────────────────────────────────────┐
│                        CONFIGURACIÓN                          │
│  config.py: Parámetros globales compartidos por todos         │
└───────────────────────────────────────────────────────────────┘
                              ▲
                              │ read/write
┌─────────────────────────────┴─────────────────────────────────┐
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐    │
│  │                 INTERFAZ GRÁFICA                      │    │
│  │  gui.py                                               │    │
│  │  ┌─────────────────────────────────────────────┐     │    │
│  │  │ StereoAppTkinter (Main Window)              │     │    │
│  │  │ - Layout Tkinter                            │     │    │
│  │  │ - Controles de parámetros                   │     │    │
│  │  │ - Botones (Pausar/Reanudar/Mapa 3D)        │     │    │
│  │  └───────────────┬─────────────────────────────┘     │    │
│  │                  │ spawn                             │    │
│  │  ┌───────────────▼─────────────────────────────┐     │    │
│  │  │ ProcesadorEstereoThread                     │     │    │
│  │  │ ┌─────────────────────────────────────┐     │     │    │
│  │  │ │     Pipeline de Procesamiento       │     │     │    │
│  │  │ └─────────────────────────────────────┘     │     │    │
│  │  └───────────────┬─────────────────────────────┘     │    │
│  └──────────────────┼───────────────────────────────────┘    │
│                     │                                          │
└─────────────────────┼──────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌────▼────┐ ┌─────▼──────┐
│ PROCESAMIENTO│ │ TRACKING│ │ DETECCIÓN  │
│   ESTÉREO    │ │         │ │   YOLO     │
└───────┬──────┘ └────┬────┘ └─────┬──────┘
        │             │             │
┌───────▼──────────────▼─────────────▼──────┐
│         MÓDULOS DE PROCESAMIENTO          │
│                                            │
│  ┌──────────────┐  ┌──────────────┐       │
│  │  stereo_     │  │  yolo_       │       │
│  │ processing   │  │ tracker      │       │
│  │              │  │              │       │
│  │ - proc_seg   │  │ - YOLOv11    │       │
│  │ - proc_mesh  │  │ - BoTSORT    │       │
│  │ - get_cns    │  │ - Marcadores │       │
│  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                │
│  ┌──────▼───────┐  ┌──────▼────────┐      │
│  │  tracker     │  │  visual_      │      │
│  │ (superviven) │  │  odometry     │      │
│  │              │  │               │      │
│  │ - Asociación │  │ - Física      │      │
│  │ - Velocidad  │  │ - Trayectoria │      │
│  │ - ID único   │  │ - Marcadores  │      │
│  └──────┬───────┘  └───────┬───────┘      │
│         │                  │               │
│  ┌──────▼──────────────────▼───────┐      │
│  │         mapper                  │      │
│  │  - Transformación afín          │      │
│  │  - Pose global (X, Y, θ)        │      │
│  │  - Mapa radar 3D                │      │
│  └─────────────────────────────────┘      │
│                                            │
│  ┌──────────────┐  ┌──────────────┐       │
│  │  anomaly_    │  │  drawing     │       │
│  │  detector    │  │              │       │
│  │              │  │ - dib_mov    │       │
│  │ - Threshold  │  │ - dib_ayu    │       │
│  │ - Estadística│  │ - dib_map    │       │
│  │ - Tracking   │  │ - Vector YOLO│       │
│  └──────────────┘  └──────────────┘       │
└────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌────▼────┐ ┌─────▼──────┐
│  UTILIDADES  │ │HARDWARE │ │VISUALIZA   │
│              │ │         │ │   CIÓN     │
│ utils.py     │ │hardware_│ │            │
│ - open_svo   │ │optimizer│ │ Open3D     │
│ - normalize  │ │ - CUDA  │ │ Tkinter    │
│ - register   │ │ - CPU   │ │ Matplotlib │
└──────────────┘ └─────────┘ └────────────┘
```

---

## 🎯 Casos de Uso Principales

### 1. **Procesamiento de Video Estéreo**
```
Usuario → Selecciona archivo SVO/MP4
       → Configura frame inicial
       → Presiona iniciar
       
Sistema → Carga video con utils.open_svo_file()
        → Inicia ProcesadorEstereoThread
        → Para cada frame:
          - Segmenta red (stereo_processing)
          - Extrae contornos (get_cns)
          - Trackea objetos (tracker)
          - Calcula pose global (mapper)
          - Actualiza GUI
```

### 2. **Detección y Tracking YOLO**
```
Usuario → Activa "YOLO Tracking"

Sistema → yolo_tracker.track_frame()
        → Detecta bordes/nudos (YOLOv11)
        → Calcula vectores de movimiento
        → Detecta tercio central
        → visual_odometry.update()
        → Genera trayectoria verde
        → Si cruza tercio: agregar marcador
        → Dibuja en gráfico 2D
```

### 3. **Visualización 3D de Trayectorias**
```
Usuario → Presiona botón "🗺️ Mapa 3D"

Sistema → Lee odometria_yolo.json
        → Lee odometria_supervivencia.json
        → Lee yolo_markers
        → Crea geometrías Open3D:
          - Cilindros para líneas
          - Esferas para puntos
          - Marcadores con palos
        → Amplifica x10
        → Abre visor Open3D
```

### 4. **Detección y Reporte de Daños**
```
Usuario → Procesa video

Sistema → anomaly_detector.detect()
        → Para cada daño confirmado:
          - Agregar a damage_log
          - Dibujar en frame
        
Usuario → Presiona "💾 Guardar Reporte"

Sistema → Genera CSV con daños
        → Guarda imagen del mapa radar
        → Exporta timestamp en nombre
```

### 5. **Ajuste de Parámetros en Tiempo Real**
```
Usuario → Mueve slider "Distancia Umbral"

Sistema → _update_config_slider()
        → config.UMB_DIST = nuevo_valor
        → Thread detecta cambio
        → tracker.update_config()
        → Aplica inmediatamente
```

---

## 🔧 Dependencias entre Módulos

### Dependencias Críticas:
```
main.py
  ├── config.py (REQUERIDO)
  ├── gui.py (REQUERIDO)
  └── hardware_optimizer.py (REQUERIDO)

gui.py
  ├── config.py (REQUERIDO)
  ├── stereo_processing.py (REQUERIDO)
  ├── tracker.py (REQUERIDO)
  ├── mapper.py (REQUERIDO)
  ├── drawing.py (REQUERIDO)
  ├── yolo_tracker.py (REQUERIDO)
  ├── visual_odometry.py (REQUERIDO)
  ├── anomaly_detector.py (REQUERIDO)
  ├── utils.py (REQUERIDO)
  └── Open3D (OPCIONAL - solo para visualización 3D)

stereo_processing.py
  ├── config.py (REQUERIDO)
  └── hardware_optimizer.py (OPCIONAL - aceleración)

yolo_tracker.py
  ├── config.py (REQUERIDO)
  ├── Ultralytics YOLO (REQUERIDO)
  └── PyTorch (REQUERIDO)

visual_odometry.py
  ├── config.py (REQUERIDO)
  └── NumPy (REQUERIDO)
```

### Dependencias Externas:
```python
# Core
numpy >= 1.20
opencv-python >= 4.5
Pillow >= 10.0

# YOLO
torch >= 2.0 (con CUDA 11.8)
ultralytics >= 8.0

# GUI
tkinter (incluido en Python)

# Visualización 3D (opcional)
open3d >= 0.16

# Video estéreo (opcional)
pyzed == 3.8
```

---

## 📈 Métricas y Rendimiento

### Tiempos de Procesamiento (estimados):
```
Frame 1920x1080 (GPU NVIDIA GTX 1060):
  - Segmentación: ~10ms
  - Matching estéreo: ~15ms
  - YOLO tracking: ~30ms (GPU)
  - Detección daños: ~20ms
  - Visualizaciones: ~5ms
  ────────────────────────────
  Total: ~80ms → ~12 FPS

Frame 1920x1080 (CPU Intel i7):
  - Segmentación: ~30ms
  - Matching estéreo: ~40ms
  - YOLO tracking: ~150ms (CPU)
  - Detección daños: ~35ms
  - Visualizaciones: ~10ms
  ────────────────────────────
  Total: ~265ms → ~4 FPS
```

### Consumo de Memoria:
```
- Aplicación base: ~200 MB
- Modelo YOLO cargado: +500 MB
- Buffer de frames: ~50 MB
- Historial tracking: ~20 MB por 100 frames
- Open3D (3D viewer): +300 MB
────────────────────────────
Total típico: ~1 GB RAM
```

### Parámetros Recomendados:
```python
# Para máxima velocidad:
SKIP_RATE = 5
PORC_MOS_INT = 30
K_UNI_SIZE = 3
K_LIMP_SIZE = 3

# Para máxima precisión:
SKIP_RATE = 1
PORC_MOS_INT = 70
K_UNI_SIZE = 5
K_LIMP_SIZE = 5

# Balance (recomendado):
SKIP_RATE = 2
PORC_MOS_INT = 50
K_UNI_SIZE = 3
K_LIMP_SIZE = 3
```

---

## 🐛 Debugging y Logs

### Sistema de Logging:
```python
# Mensajes importantes:
print("✓ ...")  # Éxito
print("⚠ ...")  # Advertencia
print("❌ ...")  # Error
print("DEBUG: ...") # Debug info
print("⭐ Marcador ...") # Marcador YOLO

# Ejemplo:
✓ Guardados 1234 frames YOLO en odometria_yolo.json
⚠ Marcador 19 saltado: frame_idx=500, traj_len=450
❌ Error guardando tracking data: [Errno 13] Permission denied
DEBUG: Intentando guardar tracking data...
⭐ Marcador 1: Borde (ID:5) en frame_idx=123, pos≈(234.5, -45.2) cm
```

### Puntos de Debug Comunes:
```python
# En gui.py - ProcesadorEstereoThread.run():
print(f"Frame {frame_counter}: processing...")

# En yolo_tracker.py:
print(f"YOLO detected: {len(detections)} objects")

# En visual_odometry.py:
if len(markers) > 0:
    print(f"🎯 Intentando dibujar {len(markers)} marcadores")
    print(f"Trayectoria tiene {len(trajectory)} puntos")

# En anomaly_detector.py:
print(f"Daños detectados: {len(confirmed_damages)}")
```

---

## 📝 Notas Finales

### Convenciones de Código:
- Nombres de clases: `PascalCase`
- Nombres de funciones: `snake_case`
- Constantes: `UPPER_SNAKE_CASE`
- Variables privadas: `_prefijo_guion_bajo`
- Comentarios en español para documentación
- Docstrings en español con formato estándar

### Formato de Coordenadas:
```python
# Imagen: (0,0) = esquina superior izquierda
#         +X = derecha, +Y = abajo

# Mundo: (0,0) = posición inicial de cámara
#        +X = derecha, +Y = adelante

# Open3D: +X = derecha, +Y = arriba, +Z = hacia observador
#         Se invierte Y al exportar: y_opencd = -y_mundo
```

### Archivos de Salida:
```
odometria_yolo.json:
  - Lista de matrices 4x4
  - Formato: [[r00, r01, r02, tx],
              [r10, r11, r12, ty],
              [r20, r21, r22, tz],
              [0,   0,   0,   1 ]]
  - Posición en metros (cm / 100)

odometria_supervivencia.json:
  - Mismo formato que YOLO
  - Basado en tracking de supervivencia

Reporte_{video}_{timestamp}_DAÑOS.csv:
  ID_Daño;Frame;X_Global_cm;Y_Global_cm;Area_px
  1;123;45,67;89,01;234,56

Reporte_{video}_{timestamp}_MAPA.png:
  - Imagen del mapa radar final
```

---

## 🚀 Inicio Rápido

### Instalación:
```bash
# 1. Clonar repositorio
git clone https://github.com/cristianurra/PDI-NET
cd PDI-NET

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar PyTorch con CUDA (opcional pero recomendado)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Instalar Open3D (opcional - para visualización 3D)
pip install open3d

# 5. Ejecutar
python script/main.py
```

### Primer Uso:
1. Seleccionar video (MP4 o SVO)
2. Configurar frame inicial (ej: 500)
3. Ajustar parámetros según necesidad
4. Activar "YOLO Tracking" si se desea
5. Presionar para iniciar procesamiento
6. Usar controles para pausar/reanudar
7. Ver mapa 3D con botón "🗺️ Mapa 3D"
8. Guardar reporte con "💾 Guardar Reporte"

---

**Última actualización**: Diciembre 2025
**Versión**: 2.0
**Autor**: PDI-NET Team
