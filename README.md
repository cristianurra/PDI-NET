# Documentación del Código de Visión Estéreo y Mapeo 2D

Este proyecto implementa un sistema de visión por computadora que combina el **seguimiento de objetos**, el **cálculo de profundidad estéreo** y la **generación de un mapa 2D simple** a partir de un video que contiene imágenes estéreo (lado a lado).

***

## 1. `config.py` (Configuraciones y Constantes)

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

## 2. `utils.py` (Funciones de necesarias)

Contiene funciones matemáticas y de ayuda usadas por los módulos de rastreo y dibujo.

| Función | Propósito |
| :--- | :--- |
| `dist(p1, p2)` | Calcula la **distancia euclidiana** entre dos puntos en el plano. |
| `depth_to_color(depth_cm)` | Mapea un valor de profundidad (en cm) a un color BGR para visualización (rojo=cercano, azul=lejano). |
| `map_trans(hist_m, m_w, m_h)` | Calcula la **escala** y el **desplazamiento** necesarios para centrar y encajar el área explorada del mapa dentro del lienzo. |
| `normalize_cell_view(...)` | Redimensiona una porción de la imagen a un tamaño estándar para guardarla como vista de una celda del mapa. |
| `register_image_to_map(...)` | Combina una nueva vista de una celda con su imagen ya existente usando un promedio ponderado. |

***

## 3. `stereo_processing.py` (Procesamiento Estéreo)

Maneja la segmentación inicial de la imagen y la lógica fundamental de emparejamiento.

| Función | Descripción |
| :--- | :--- |
| `proc_seg(frame, k_uni, k_limp)` | Realiza la **segmentación** de objetos mediante umbralización adaptativa y operaciones morfológicas (cierre y apertura) para limpiar contornos. |
| `get_cns(cns_filt, q_w, q_h, w)` | 1. **Contornos:** Encuentra contornos válidos dentro de las Regiones de Interés (ROI). 2. **Emparejamiento Estéreo:** Intenta emparejar contornos del ojo izquierdo con los del ojo derecho, aplicando la **restricción de línea de barrido** (tolerancia Y) y la **restricción de disparidad**. 3. Devuelve los centroides emparejados y su disparidad. |

***

## 4. `tracker.py` (Seguimiento de puntos)

Contiene la clase `Tracker`, que gestiona la persistencia de los objetos, su movimiento y sus propiedades.

| Clase / Método | Propósito y Lógica |
| :--- | :--- |
| `class Tracker` | Mantiene el estado de cada objeto rastreado (historial de posición, velocidad y profundidad). |
| `update_and_get(...)` | **Algoritmo de seguimiento:** Predice la posición, asocia nuevos contornos usando `UMB_DIST`, actualiza la posición, velocidad, profundidad (**$D = \frac{f \cdot B}{d}$**), y gestiona la supervivencia del objeto (detección de estático). |

***

## 5. `drawing.py` (Visualización)

Agrupa todas las funciones responsables de dibujar la información y la interfaz de usuario.

| Función | Elementos que dibuja |
| :--- | :--- |
| `dib_escala_profundidad(...)` | La barra de color lateral que muestra la escala de profundidad. |
| `dib_mov(...)` | Dibuja los puntos rastreados y estima el **vector de movimiento general de la cámara** (Odometría Visual) a partir de los objetos. Muestra la profundidad promedio y el movimiento en texto. |
| `dib_ayu(...)` | Dibuja líneas de referencia, etiquetas de "Ojo Izq/Der" y los límites de las Regiones de Interés (ROI). |
| `dib_map(...)` | Renderiza el **mapa 2D**. Dibuja las celdas visitadas con la imagen guardada y muestra la posición actual y el área de visión de la cámara. |
| `show_compuesta(...)` | Ensambla las diferentes partes (`frame` principal, segmentación, y mapa) y las muestra en una única ventana. |

***

## 6. `main.py` (Bucle Principal)

Define el flujo de ejecución por *frame* del programa.

1. **Captura y Procesamiento:** Lee el *frame*, llama a `proc_seg`, `get_cns`, y `tracker.update_and_get`.
2. **Cálculo de Odometría:** Usa el vector de movimiento de la cámara estimado y lo traduce a centímetros usando `CM_POR_PX` para actualizar la posición global $(\text{pos\_m\_x}, \text{pos\_m\_y})$.
3. **Actualización del Mapa:** La posición global se usa para determinar la celda $(\text{grid\_x}, \text{grid\_y})$ y se actualiza el diccionario `hist_celdas_vis` con la vista actual.
4. **Visualización:** Llama a todas las funciones de `drawing.py` para construir y mostrar la interfaz.
