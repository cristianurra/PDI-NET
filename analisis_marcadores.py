#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script para análisis estadístico de mediciones de marcadores
Procesa el CSV generado por el sistema de odometría visual
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

def cargar_csv_mas_reciente():
    """Carga el archivo CSV más reciente de mediciones"""
    archivos = glob.glob('mediciones_marcadores_*.csv')
    if not archivos:
        print("❌ No se encontraron archivos de mediciones")
        return None
    
    archivo_mas_reciente = max(archivos, key=lambda x: Path(x).stat().st_mtime)
    print(f"📂 Cargando: {archivo_mas_reciente}")
    return pd.read_csv(archivo_mas_reciente)

def analisis_estadistico_basico(df):
    """Realiza análisis estadístico básico"""
    print("\n" + "="*60)
    print("ANÁLISIS ESTADÍSTICO BÁSICO")
    print("="*60)
    
    print(f"\n📊 Total de mediciones: {len(df)}")
    print(f"📅 Fecha: {df['timestamp'].iloc[0] if len(df) > 0 else 'N/A'}")
    
    print("\n--- ESTADÍSTICAS DE ERROR YOLO ---")
    print(f"Media:        {df['error_yolo_cm'].mean():.2f} cm")
    print(f"Mediana:      {df['error_yolo_cm'].median():.2f} cm")
    print(f"Desv. Std:    {df['error_yolo_cm'].std():.2f} cm")
    print(f"Mínimo:       {df['error_yolo_cm'].min():.2f} cm")
    print(f"Máximo:       {df['error_yolo_cm'].max():.2f} cm")
    print(f"Error total:  {df['error_yolo_cm'].sum():.2f} cm")
    
    print("\n--- ESTADÍSTICAS DE ERROR SUPERVIVENCIA ---")
    print(f"Media:        {df['error_supervivencia_cm'].mean():.2f} cm")
    print(f"Mediana:      {df['error_supervivencia_cm'].median():.2f} cm")
    print(f"Desv. Std:    {df['error_supervivencia_cm'].std():.2f} cm")
    print(f"Mínimo:       {df['error_supervivencia_cm'].min():.2f} cm")
    print(f"Máximo:       {df['error_supervivencia_cm'].max():.2f} cm")
    print(f"Error total:  {df['error_supervivencia_cm'].sum():.2f} cm")
    
    print("\n--- COMPARACIÓN DE ALGORITMOS ---")
    mejor_conteo = df['mejor_algoritmo'].value_counts()
    print(mejor_conteo)
    
    # Calcular porcentaje de error promedio
    error_pct_yolo = (df['error_yolo_cm'] / df['distancia_real_cm'] * 100).mean()
    error_pct_superv = (df['error_supervivencia_cm'] / df['distancia_real_cm'] * 100).mean()
    
    print(f"\n📈 Error promedio YOLO: {error_pct_yolo:.2f}%")
    print(f"📈 Error promedio Supervivencia: {error_pct_superv:.2f}%")

def analisis_por_segmento(df):
    """Analiza el error acumulado por segmentos"""
    print("\n" + "="*60)
    print("ANÁLISIS POR SEGMENTO")
    print("="*60)
    
    df['error_acumulado_yolo'] = df['error_yolo_cm'].cumsum()
    df['error_acumulado_superv'] = df['error_supervivencia_cm'].cumsum()
    
    print("\nError acumulado por marcador:")
    print(df[['marcador_desde', 'marcador_hasta', 
             'error_yolo_cm', 'error_acumulado_yolo',
             'error_supervivencia_cm', 'error_acumulado_superv']].to_string(index=False))

def generar_graficos(df):
    """Genera visualizaciones de los datos"""
    print("\n" + "="*60)
    print("GENERANDO GRÁFICOS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Precisión de Odometría Visual', fontsize=16, fontweight='bold')
    
    # 1. Comparación de errores por marcador
    ax1 = axes[0, 0]
    x = range(len(df))
    ax1.plot(x, df['error_yolo_cm'], marker='o', label='YOLO', linewidth=2)
    ax1.plot(x, df['error_supervivencia_cm'], marker='s', label='Supervivencia', linewidth=2)
    ax1.axhline(y=df['error_yolo_cm'].mean(), color='blue', linestyle='--', alpha=0.5, label='Media YOLO')
    ax1.axhline(y=df['error_supervivencia_cm'].mean(), color='orange', linestyle='--', alpha=0.5, label='Media Superv.')
    ax1.set_xlabel('Segmento (Marcador N -> N+1)')
    ax1.set_ylabel('Error absoluto (cm)')
    ax1.set_title('Error por Segmento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error acumulado
    ax2 = axes[0, 1]
    df['error_acum_yolo'] = df['error_yolo_cm'].cumsum()
    df['error_acum_superv'] = df['error_supervivencia_cm'].cumsum()
    ax2.plot(x, df['error_acum_yolo'], marker='o', label='YOLO', linewidth=2)
    ax2.plot(x, df['error_acum_superv'], marker='s', label='Supervivencia', linewidth=2)
    ax2.set_xlabel('Segmento')
    ax2.set_ylabel('Error acumulado (cm)')
    ax2.set_title('Error Acumulado')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot de errores
    ax3 = axes[1, 0]
    data_box = [df['error_yolo_cm'], df['error_supervivencia_cm']]
    bp = ax3.boxplot(data_box, labels=['YOLO', 'Supervivencia'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Error (cm)')
    ax3.set_title('Distribución de Errores')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distancia medida vs real
    ax4 = axes[1, 1]
    ax4.scatter(df['distancia_real_cm'], df['distancia_yolo_cm'], 
               alpha=0.6, s=100, label='YOLO', marker='o')
    ax4.scatter(df['distancia_real_cm'], df['distancia_supervivencia_cm'], 
               alpha=0.6, s=100, label='Supervivencia', marker='s')
    
    # Línea de referencia perfecta
    max_dist = max(df['distancia_real_cm'].max(), 
                  df['distancia_yolo_cm'].max(), 
                  df['distancia_supervivencia_cm'].max())
    ax4.plot([0, max_dist], [0, max_dist], 'k--', alpha=0.3, label='Ideal')
    
    ax4.set_xlabel('Distancia Real (cm)')
    ax4.set_ylabel('Distancia Medida (cm)')
    ax4.set_title('Precisión de Medición')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar figura
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'analisis_odometria_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico guardado: {filename}")
    
    plt.show()

def exportar_resumen(df):
    """Exporta un resumen en formato texto"""
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f'resumen_analisis_{timestamp}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("RESUMEN DE ANÁLISIS DE ODOMETRÍA VISUAL\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total de mediciones: {len(df)}\n")
        f.write(f"Fecha de análisis: {pd.Timestamp.now()}\n\n")
        
        f.write("YOLO:\n")
        f.write(f"  Error promedio: {df['error_yolo_cm'].mean():.2f} cm\n")
        f.write(f"  Error total: {df['error_yolo_cm'].sum():.2f} cm\n")
        f.write(f"  Desviación estándar: {df['error_yolo_cm'].std():.2f} cm\n\n")
        
        f.write("Supervivencia:\n")
        f.write(f"  Error promedio: {df['error_supervivencia_cm'].mean():.2f} cm\n")
        f.write(f"  Error total: {df['error_supervivencia_cm'].sum():.2f} cm\n")
        f.write(f"  Desviación estándar: {df['error_supervivencia_cm'].std():.2f} cm\n\n")
        
        f.write("Mejor algoritmo por medición:\n")
        f.write(df['mejor_algoritmo'].value_counts().to_string() + "\n")
    
    print(f"✅ Resumen exportado: {filename}")

def main():
    """Función principal"""
    print("\n" + "🔬"*30)
    print("ANÁLISIS ESTADÍSTICO DE ODOMETRÍA VISUAL")
    print("🔬"*30 + "\n")
    
    # Cargar datos
    df = cargar_csv_mas_reciente()
    if df is None or len(df) == 0:
        print("❌ No hay datos para analizar")
        return
    
    # Realizar análisis
    analisis_estadistico_basico(df)
    analisis_por_segmento(df)
    
    # Generar visualizaciones
    generar_graficos(df)
    
    # Exportar resumen
    exportar_resumen(df)
    
    print("\n" + "✅"*30)
    print("ANÁLISIS COMPLETADO")
    print("✅"*30 + "\n")

if __name__ == "__main__":
    main()
