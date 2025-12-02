"""
Módulo de optimización de hardware para el sistema de procesamiento estéreo.
Detecta y configura CUDA, multiprocessing y optimizaciones de OpenCV.
"""

import cv2
import numpy as np
import multiprocessing as mp
import os
from typing import Optional, Tuple


class HardwareOptimizer:
    """
    Clase para detectar y configurar optimizaciones de hardware disponibles.
    """
    
    def __init__(self):
        self.cuda_available = False
        self.cuda_device_count = 0
        self.cpu_count = mp.cpu_count()
        self.opencv_cuda_enabled = False
        
        self._detect_capabilities()
        self._configure_opencv()
    
    def _detect_capabilities(self):
        """Detecta capacidades de hardware disponibles."""
        # Detectar CUDA en PyTorch (más común y mejor soporte)
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
            
            if self.cuda_available:
                self.cuda_device_count = torch.cuda.device_count()
                print(f"✓ CUDA detectado (PyTorch): {self.cuda_device_count} dispositivo(s)")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA Version: {torch.version.cuda}")
                print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
            else:
                print("✗ CUDA no disponible en PyTorch")
                self.cuda_device_count = 0
        except ImportError:
            print("✗ PyTorch no instalado, CUDA no disponible")
            self.cuda_available = False
            self.cuda_device_count = 0
        except Exception as e:
            print(f"✗ Error detectando CUDA: {e}")
            self.cuda_available = False
            self.cuda_device_count = 0
        
        # Detectar CUDA en OpenCV (opcional, menos común)
        try:
            opencv_cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if opencv_cuda_count > 0:
                self.opencv_cuda_enabled = True
                print(f"✓ OpenCV CUDA también disponible: {opencv_cuda_count} dispositivo(s)")
            else:
                self.opencv_cuda_enabled = False
                print("  OpenCV sin soporte CUDA (usando CPU para operaciones OpenCV)")
        except:
            self.opencv_cuda_enabled = False
            print("  OpenCV sin soporte CUDA (usando CPU para operaciones OpenCV)")
        
        print(f"✓ CPUs disponibles: {self.cpu_count}")
    
    def _configure_opencv(self):
        """Configura OpenCV para usar optimizaciones disponibles."""
        # Habilitar optimizaciones automáticas de OpenCV
        cv2.setUseOptimized(True)
        
        # Configurar número de threads de OpenCV
        num_threads = max(1, self.cpu_count - 1)  # Dejar 1 CPU libre para GUI
        cv2.setNumThreads(num_threads)
        
        print(f"✓ OpenCV configurado con {num_threads} threads")
        print(f"✓ Optimizaciones OpenCV: {cv2.useOptimized()}")
    
    def get_optimal_threads(self) -> int:
        """Retorna el número óptimo de threads para procesamiento paralelo."""
        return max(1, self.cpu_count - 1)
    
    def is_cuda_available(self) -> bool:
        """Retorna True si CUDA está disponible."""
        return self.cuda_available


class CUDAProcessor:
    """
    Clase para operaciones aceleradas por CUDA cuando está disponible.
    Usa PyTorch CUDA para operaciones que OpenCV no soporta.
    """
    
    def __init__(self, use_cuda: bool = True):
        # Verificar si PyTorch tiene CUDA disponible
        self.use_cuda = use_cuda
        self.pytorch_cuda_available = False
        self.opencv_cuda_available = False
        
        if use_cuda:
            try:
                import torch
                self.pytorch_cuda_available = torch.cuda.is_available()
            except ImportError:
                pass
            
            try:
                self.opencv_cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            except:
                pass
        
        self.use_cuda = self.use_cuda and (self.pytorch_cuda_available or self.opencv_cuda_available)
        self.gpu_mat_cache = {}
        
        if self.use_cuda:
            if self.pytorch_cuda_available:
                print("✓ Procesador CUDA inicializado (PyTorch)")
            if self.opencv_cuda_available:
                print("✓ Procesador CUDA inicializado (OpenCV)")
        else:
            print("  Usando procesamiento CPU (CUDA no disponible)")
    
    def adaptive_threshold_cuda(self, gray: np.ndarray, max_value: int = 255, 
                                adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                threshold_type: int = cv2.THRESH_BINARY_INV,
                                block_size: int = 15, c: int = 1) -> np.ndarray:
        """
        Umbralización adaptativa con aceleración CUDA si está disponible.
        """
        if not self.use_cuda:
            return cv2.adaptiveThreshold(gray, max_value, adaptive_method, 
                                        threshold_type, block_size, c)
        
        try:
            # Subir a GPU
            gpu_gray = cv2.cuda_GpuMat()
            gpu_gray.upload(gray)
            
            # Procesar en GPU (OpenCV CUDA tiene limitaciones en adaptive threshold)
            # Fallback a CPU si no está disponible
            result = cv2.adaptiveThreshold(gray, max_value, adaptive_method, 
                                          threshold_type, block_size, c)
            return result
        except Exception as e:
            # Fallback a CPU
            return cv2.adaptiveThreshold(gray, max_value, adaptive_method, 
                                        threshold_type, block_size, c)
    
    def gaussian_blur_cuda(self, img: np.ndarray, ksize: Tuple[int, int], 
                          sigma: float = 0) -> np.ndarray:
        """
        Desenfoque gaussiano con aceleración CUDA si está disponible.
        """
        if not self.opencv_cuda_available:
            return cv2.GaussianBlur(img, ksize, sigma)
        
        try:
            # Subir a GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            # Crear filtro gaussiano GPU
            gaussian_filter = cv2.cuda.createGaussianFilter(
                gpu_img.type(), -1, ksize, sigma
            )
            
            # Aplicar filtro
            gpu_result = gaussian_filter.apply(gpu_img)
            
            # Descargar resultado
            return gpu_result.download()
        except Exception as e:
            # Fallback a CPU
            return cv2.GaussianBlur(img, ksize, sigma)
    
    def morphology_cuda(self, img: np.ndarray, op: int, kernel: np.ndarray) -> np.ndarray:
        """
        Operaciones morfológicas con aceleración CUDA si está disponible.
        """
        if not self.opencv_cuda_available:
            return cv2.morphologyEx(img, op, kernel)
        
        try:
            # Subir a GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            # Crear filtro morfológico GPU
            morph_filter = cv2.cuda.createMorphologyFilter(
                op, gpu_img.type(), kernel
            )
            
            # Aplicar filtro
            gpu_result = morph_filter.apply(gpu_img)
            
            # Descargar resultado
            return gpu_result.download()
        except Exception as e:
            # Fallback a CPU
            return cv2.morphologyEx(img, op, kernel)
    
    def resize_cuda(self, img: np.ndarray, dsize: Tuple[int, int], 
                   interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Redimensionamiento con aceleración CUDA si está disponible.
        """
        if not self.opencv_cuda_available:
            return cv2.resize(img, dsize, interpolation=interpolation)
        
        try:
            # Subir a GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            # Redimensionar en GPU
            gpu_result = cv2.cuda.resize(gpu_img, dsize, interpolation=interpolation)
            
            # Descargar resultado
            return gpu_result.download()
        except Exception as e:
            # Fallback a CPU
            return cv2.resize(img, dsize, interpolation=interpolation)
    
    def cvt_color_cuda(self, img: np.ndarray, code: int) -> np.ndarray:
        """
        Conversión de color con aceleración CUDA si está disponible.
        """
        if not self.opencv_cuda_available:
            return cv2.cvtColor(img, code)
        
        try:
            # Subir a GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            
            # Convertir en GPU
            gpu_result = cv2.cuda.cvtColor(gpu_img, code)
            
            # Descargar resultado
            return gpu_result.download()
        except Exception as e:
            # Fallback a CPU
            return cv2.cvtColor(img, code)


# Instancia global del optimizador
_hardware_optimizer = None
_cuda_processor = None


def get_hardware_optimizer() -> HardwareOptimizer:
    """Retorna la instancia global del optimizador de hardware."""
    global _hardware_optimizer
    if _hardware_optimizer is None:
        _hardware_optimizer = HardwareOptimizer()
    return _hardware_optimizer


def get_cuda_processor() -> CUDAProcessor:
    """Retorna la instancia global del procesador CUDA."""
    global _cuda_processor
    if _cuda_processor is None:
        optimizer = get_hardware_optimizer()
        _cuda_processor = CUDAProcessor(use_cuda=optimizer.is_cuda_available())
    return _cuda_processor


def initialize_hardware_optimization():
    """Inicializa las optimizaciones de hardware al inicio del programa."""
    print("\n" + "="*60)
    print("INICIALIZANDO OPTIMIZACIONES DE HARDWARE")
    print("="*60)
    
    optimizer = get_hardware_optimizer()
    cuda_proc = get_cuda_processor()
    
    print("="*60 + "\n")
    
    return optimizer, cuda_proc
