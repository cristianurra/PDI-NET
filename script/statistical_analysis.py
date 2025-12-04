#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo de análisis estadístico para el sistema de odometría visual.
Ejecuta automáticamente al finalizar una sesión de tracking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict
import json
from scipy import stats


class AnalizadorEstadistico:
    """
    Clase para realizar análisis estadístico completo de las mediciones de odometría.
    """
    
    def __init__(self, session_path: str):
        """
        Inicializa el analizador estadístico.
        
        Args:
            session_path: Ruta de la carpeta de sesión (report/sesion_YYYYMMDD_HHMMSS/)
        """
        self.session_path = Path(session_path)
        self.analysis_path = self.session_path / "analisis"
        self.analysis_path.mkdir(exist_ok=True)
        
        # Archivos de entrada
        self.csv_file = self.session_path / "mediciones_marcadores.csv"
        self.yolo_json = self.session_path / "odometria_yolo.json"
        self.superv_json = self.session_path / "odometria_supervivencia.json"
        
        self.df = None
        self.metricas = {}
        
    def cargar_datos(self) -> bool:
        """Carga el CSV de mediciones."""
        try:
            if not self.csv_file.exists():
                print(f"⚠️ No se encontró archivo de mediciones: {self.csv_file}")
                return False
            
            self.df = pd.read_csv(self.csv_file)
            if len(self.df) == 0:
                print("⚠️ El archivo de mediciones está vacío")
                return False
            
            print(f"✓ Cargadas {len(self.df)} mediciones desde {self.csv_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return False
    
    def calcular_metricas(self) -> Dict:
        """
        Calcula métricas estadísticas completas.
        
        Returns:
            Diccionario con todas las métricas calculadas
        """
        if self.df is None or len(self.df) == 0:
            return {}
        
        # Distancia de referencia (100 cm entre marcadores)
        distancia_ref = self.df['distancia_real_cm'].iloc[0] if len(self.df) > 0 else 100.0
        
        # ===== MÉTRICAS YOLO =====
        errores_yolo = self.df['error_yolo_cm'].values
        distancias_yolo = self.df['distancia_yolo_cm'].values
        
        # Error cuadrático medio (MSE) y raíz (RMSE)
        mse_yolo = np.mean(errores_yolo ** 2)
        rmse_yolo = np.sqrt(mse_yolo)
        
        # Error absoluto medio (MAE)
        mae_yolo = np.mean(np.abs(errores_yolo))
        
        # Error máximo
        max_error_yolo = np.max(np.abs(errores_yolo))
        
        # Error porcentual
        error_pct_yolo = (mae_yolo / distancia_ref) * 100
        
        # Desviación estándar
        std_yolo = np.std(errores_yolo)
        
        # Coeficiente de variación
        cv_yolo = (std_yolo / np.mean(distancias_yolo)) * 100 if np.mean(distancias_yolo) > 0 else 0
        
        # Error acumulado
        error_acum_yolo = np.sum(np.abs(errores_yolo))
        
        # ===== MÉTRICAS SUPERVIVENCIA =====
        errores_superv = self.df['error_supervivencia_cm'].values
        distancias_superv = self.df['distancia_supervivencia_cm'].values
        
        # Error cuadrático medio (MSE) y raíz (RMSE)
        mse_superv = np.mean(errores_superv ** 2)
        rmse_superv = np.sqrt(mse_superv)
        
        # Error absoluto medio (MAE)
        mae_superv = np.mean(np.abs(errores_superv))
        
        # Error máximo
        max_error_superv = np.max(np.abs(errores_superv))
        
        # Error porcentual
        error_pct_superv = (mae_superv / distancia_ref) * 100
        
        # Desviación estándar
        std_superv = np.std(errores_superv)
        
        # Coeficiente de variación
        cv_superv = (std_superv / np.mean(distancias_superv)) * 100 if np.mean(distancias_superv) > 0 else 0
        
        # Error acumulado
        error_acum_superv = np.sum(np.abs(errores_superv))
        
        # ===== CORRELACIÓN =====
        # Correlación de Pearson entre distancia medida y real
        corr_yolo, pval_yolo = stats.pearsonr(distancias_yolo, self.df['distancia_real_cm'].values)
        corr_superv, pval_superv = stats.pearsonr(distancias_superv, self.df['distancia_real_cm'].values)
        
        # Correlación entre ambos métodos
        corr_metodos, pval_metodos = stats.pearsonr(distancias_yolo, distancias_superv)
        
        # ===== R² (Coeficiente de determinación) =====
        ss_res_yolo = np.sum((self.df['distancia_real_cm'].values - distancias_yolo) ** 2)
        ss_tot_yolo = np.sum((self.df['distancia_real_cm'].values - np.mean(self.df['distancia_real_cm'].values)) ** 2)
        r2_yolo = 1 - (ss_res_yolo / ss_tot_yolo) if ss_tot_yolo > 0 else 0
        
        ss_res_superv = np.sum((self.df['distancia_real_cm'].values - distancias_superv) ** 2)
        ss_tot_superv = np.sum((self.df['distancia_real_cm'].values - np.mean(self.df['distancia_real_cm'].values)) ** 2)
        r2_superv = 1 - (ss_res_superv / ss_tot_superv) if ss_tot_superv > 0 else 0
        
        # ===== MEJOR ALGORITMO =====
        mejor_conteo = self.df['mejor_algoritmo'].value_counts().to_dict()
        
        # Guardar todas las métricas
        self.metricas = {
            'yolo': {
                'mse': mse_yolo,
                'rmse': rmse_yolo,
                'mae': mae_yolo,
                'max_error': max_error_yolo,
                'error_pct': error_pct_yolo,
                'std': std_yolo,
                'cv': cv_yolo,
                'error_acumulado': error_acum_yolo,
                'correlacion': corr_yolo,
                'pvalue_corr': pval_yolo,
                'r2': r2_yolo
            },
            'supervivencia': {
                'mse': mse_superv,
                'rmse': rmse_superv,
                'mae': mae_superv,
                'max_error': max_error_superv,
                'error_pct': error_pct_superv,
                'std': std_superv,
                'cv': cv_superv,
                'error_acumulado': error_acum_superv,
                'correlacion': corr_superv,
                'pvalue_corr': pval_superv,
                'r2': r2_superv
            },
            'comparacion': {
                'correlacion_entre_metodos': corr_metodos,
                'pvalue_metodos': pval_metodos,
                'mejor_conteo': mejor_conteo,
                'distancia_referencia': distancia_ref,
                'num_mediciones': len(self.df)
            }
        }
        
        return self.metricas
    
    def imprimir_metricas(self):
        """Imprime las métricas en consola de forma legible."""
        if not self.metricas:
            return
        
        print("\n" + "="*70)
        print("📊 ANÁLISIS ESTADÍSTICO COMPLETO DE ODOMETRÍA VISUAL")
        print("="*70)
        
        print(f"\n📁 Sesión: {self.session_path.name}")
        print(f"📏 Distancia de referencia: {self.metricas['comparacion']['distancia_referencia']:.1f} cm")
        print(f"📈 Total de mediciones: {self.metricas['comparacion']['num_mediciones']}")
        
        # YOLO
        print("\n" + "-"*70)
        print("🟢 YOLO - Métricas de Error")
        print("-"*70)
        y = self.metricas['yolo']
        print(f"  MSE (Error Cuadrático Medio):     {y['mse']:8.2f} cm²")
        print(f"  RMSE (Raíz MSE):                  {y['rmse']:8.2f} cm")
        print(f"  MAE (Error Absoluto Medio):       {y['mae']:8.2f} cm")
        print(f"  Error Máximo:                     {y['max_error']:8.2f} cm")
        print(f"  Error Porcentual:                 {y['error_pct']:8.2f} %")
        print(f"  Desviación Estándar:              {y['std']:8.2f} cm")
        print(f"  Coeficiente de Variación:         {y['cv']:8.2f} %")
        print(f"  Error Acumulado Total:            {y['error_acumulado']:8.2f} cm")
        print(f"  Correlación con distancia real:   {y['correlacion']:8.4f} (p={y['pvalue_corr']:.4f})")
        print(f"  R² (Coef. determinación):         {y['r2']:8.4f}")
        
        # Supervivencia
        print("\n" + "-"*70)
        print("🔵 SUPERVIVENCIA - Métricas de Error")
        print("-"*70)
        s = self.metricas['supervivencia']
        print(f"  MSE (Error Cuadrático Medio):     {s['mse']:8.2f} cm²")
        print(f"  RMSE (Raíz MSE):                  {s['rmse']:8.2f} cm")
        print(f"  MAE (Error Absoluto Medio):       {s['mae']:8.2f} cm")
        print(f"  Error Máximo:                     {s['max_error']:8.2f} cm")
        print(f"  Error Porcentual:                 {s['error_pct']:8.2f} %")
        print(f"  Desviación Estándar:              {s['std']:8.2f} cm")
        print(f"  Coeficiente de Variación:         {s['cv']:8.2f} %")
        print(f"  Error Acumulado Total:            {s['error_acumulado']:8.2f} cm")
        print(f"  Correlación con distancia real:   {s['correlacion']:8.4f} (p={s['pvalue_corr']:.4f})")
        print(f"  R² (Coef. determinación):         {s['r2']:8.4f}")
        
        # Comparación
        print("\n" + "-"*70)
        print("⚖️  COMPARACIÓN ENTRE MÉTODOS")
        print("-"*70)
        c = self.metricas['comparacion']
        print(f"  Correlación YOLO ↔ Supervivencia: {c['correlacion_entre_metodos']:8.4f} (p={c['pvalue_metodos']:.4f})")
        print(f"\n  Mejor algoritmo por segmento:")
        for metodo, conteo in c['mejor_conteo'].items():
            porcentaje = (conteo / c['num_mediciones']) * 100
            print(f"    {metodo:20s}: {conteo:3d} veces ({porcentaje:5.1f}%)")
        
        # Conclusión
        print("\n" + "-"*70)
        print("🎯 CONCLUSIÓN")
        print("-"*70)
        mejor_rmse = "YOLO" if y['rmse'] < s['rmse'] else "Supervivencia"
        mejor_mae = "YOLO" if y['mae'] < s['mae'] else "Supervivencia"
        mejor_corr = "YOLO" if y['correlacion'] > s['correlacion'] else "Supervivencia"
        
        print(f"  Menor RMSE:           {mejor_rmse}")
        print(f"  Menor MAE:            {mejor_mae}")
        print(f"  Mayor Correlación:    {mejor_corr}")
        print("="*70 + "\n")
    
    def generar_graficos(self):
        """Genera gráficos estadísticos avanzados."""
        if self.df is None or len(self.df) == 0:
            return
        
        # Configurar estilo
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 10
        
        # Crear figura con 6 subplots (3x2)
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # ===== 1. Error por segmento =====
        ax1 = fig.add_subplot(gs[0, :2])
        x = range(len(self.df))
        labels = [f"{r['marcador_desde']}→{r['marcador_hasta']}" for _, r in self.df.iterrows()]
        
        ax1.plot(x, self.df['error_yolo_cm'], 'o-', color='green', linewidth=2, 
                markersize=8, label='YOLO', alpha=0.7)
        ax1.plot(x, self.df['error_supervivencia_cm'], 's-', color='blue', linewidth=2, 
                markersize=8, label='Supervivencia', alpha=0.7)
        
        # Líneas de media
        ax1.axhline(y=self.metricas['yolo']['mae'], color='green', linestyle='--', 
                   alpha=0.4, label=f'MAE YOLO: {self.metricas["yolo"]["mae"]:.1f} cm')
        ax1.axhline(y=self.metricas['supervivencia']['mae'], color='blue', linestyle='--', 
                   alpha=0.4, label=f'MAE Superv.: {self.metricas["supervivencia"]["mae"]:.1f} cm')
        
        ax1.set_xlabel('Segmento', fontweight='bold')
        ax1.set_ylabel('Error Absoluto (cm)', fontweight='bold')
        ax1.set_title('Error de Medición por Segmento', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # ===== 2. Distribución de errores (Boxplot + Violin) =====
        ax2 = fig.add_subplot(gs[0, 2])
        data_box = [self.df['error_yolo_cm'], self.df['error_supervivencia_cm']]
        
        # Violin plot
        parts = ax2.violinplot(data_box, positions=[1, 2], showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.6)
        
        # Boxplot superpuesto
        bp = ax2.boxplot(data_box, positions=[1, 2], widths=0.3, patch_artist=True,
                        boxprops=dict(alpha=0.4))
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightblue')
        
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(['YOLO', 'Supervivencia'])
        ax2.set_ylabel('Error (cm)', fontweight='bold')
        ax2.set_title('Distribución de Errores', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ===== 3. Error acumulado =====
        ax3 = fig.add_subplot(gs[1, 0])
        error_acum_yolo = np.cumsum(np.abs(self.df['error_yolo_cm']))
        error_acum_superv = np.cumsum(np.abs(self.df['error_supervivencia_cm']))
        
        ax3.plot(x, error_acum_yolo, 'o-', color='green', linewidth=2, 
                markersize=6, label='YOLO', alpha=0.7)
        ax3.plot(x, error_acum_superv, 's-', color='blue', linewidth=2, 
                markersize=6, label='Supervivencia', alpha=0.7)
        
        ax3.set_xlabel('Segmento', fontweight='bold')
        ax3.set_ylabel('Error Acumulado (cm)', fontweight='bold')
        ax3.set_title('Error Acumulado por Segmento', fontsize=12, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        # ===== 4. Distancia medida vs real (Scatter + línea ideal) =====
        ax4 = fig.add_subplot(gs[1, 1])
        
        ax4.scatter(self.df['distancia_real_cm'], self.df['distancia_yolo_cm'], 
                   s=120, alpha=0.6, color='green', marker='o', label='YOLO', edgecolors='darkgreen')
        ax4.scatter(self.df['distancia_real_cm'], self.df['distancia_supervivencia_cm'], 
                   s=120, alpha=0.6, color='blue', marker='s', label='Supervivencia', edgecolors='darkblue')
        
        # Línea ideal (y=x)
        max_dist = max(self.df['distancia_real_cm'].max(), 
                      self.df['distancia_yolo_cm'].max(), 
                      self.df['distancia_supervivencia_cm'].max())
        min_dist = min(self.df['distancia_real_cm'].min(), 
                      self.df['distancia_yolo_cm'].min(), 
                      self.df['distancia_supervivencia_cm'].min())
        ax4.plot([min_dist, max_dist], [min_dist, max_dist], 'k--', alpha=0.4, 
                linewidth=2, label='Medición Perfecta')
        
        ax4.set_xlabel('Distancia Real (cm)', fontweight='bold')
        ax4.set_ylabel('Distancia Medida (cm)', fontweight='bold')
        ax4.set_title('Precisión de Medición', fontsize=12, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal', adjustable='box')
        
        # Añadir R²
        ax4.text(0.05, 0.95, f'R² YOLO: {self.metricas["yolo"]["r2"]:.4f}\nR² Superv.: {self.metricas["supervivencia"]["r2"]:.4f}',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
        
        # ===== 5. Histograma de residuales =====
        ax5 = fig.add_subplot(gs[1, 2])
        
        bins = np.linspace(min(self.df['error_yolo_cm'].min(), self.df['error_supervivencia_cm'].min()),
                          max(self.df['error_yolo_cm'].max(), self.df['error_supervivencia_cm'].max()),
                          15)
        
        ax5.hist(self.df['error_yolo_cm'], bins=bins, alpha=0.5, color='green', 
                label='YOLO', edgecolor='darkgreen')
        ax5.hist(self.df['error_supervivencia_cm'], bins=bins, alpha=0.5, color='blue', 
                label='Supervivencia', edgecolor='darkblue')
        
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Error = 0')
        ax5.set_xlabel('Error (cm)', fontweight='bold')
        ax5.set_ylabel('Frecuencia', fontweight='bold')
        ax5.set_title('Distribución de Residuales', fontsize=12, fontweight='bold')
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # ===== 6. Correlación entre métodos =====
        ax6 = fig.add_subplot(gs[2, 0])
        
        ax6.scatter(self.df['distancia_yolo_cm'], self.df['distancia_supervivencia_cm'],
                   s=120, alpha=0.6, color='purple', edgecolors='darkviolet')
        
        # Línea de tendencia
        z = np.polyfit(self.df['distancia_yolo_cm'], self.df['distancia_supervivencia_cm'], 1)
        p = np.poly1d(z)
        ax6.plot(self.df['distancia_yolo_cm'], p(self.df['distancia_yolo_cm']), 
                "r--", alpha=0.8, linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax6.set_xlabel('Distancia YOLO (cm)', fontweight='bold')
        ax6.set_ylabel('Distancia Supervivencia (cm)', fontweight='bold')
        ax6.set_title('Correlación entre Métodos', fontsize=12, fontweight='bold')
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)
        
        # Añadir correlación
        corr = self.metricas['comparacion']['correlacion_entre_metodos']
        ax6.text(0.05, 0.95, f'Correlación: {corr:.4f}',
                transform=ax6.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7), fontsize=10)
        
        # ===== 7. Tabla de métricas comparativas =====
        ax7 = fig.add_subplot(gs[2, 1:])
        ax7.axis('tight')
        ax7.axis('off')
        
        # Crear tabla
        metricas_tabla = [
            ['Métrica', 'YOLO', 'Supervivencia', 'Mejor'],
            ['MSE (cm²)', f"{self.metricas['yolo']['mse']:.2f}", 
             f"{self.metricas['supervivencia']['mse']:.2f}",
             '✓' if self.metricas['yolo']['mse'] < self.metricas['supervivencia']['mse'] else ''],
            ['RMSE (cm)', f"{self.metricas['yolo']['rmse']:.2f}", 
             f"{self.metricas['supervivencia']['rmse']:.2f}",
             '✓' if self.metricas['yolo']['rmse'] < self.metricas['supervivencia']['rmse'] else ''],
            ['MAE (cm)', f"{self.metricas['yolo']['mae']:.2f}", 
             f"{self.metricas['supervivencia']['mae']:.2f}",
             '✓' if self.metricas['yolo']['mae'] < self.metricas['supervivencia']['mae'] else ''],
            ['Error Máx (cm)', f"{self.metricas['yolo']['max_error']:.2f}", 
             f"{self.metricas['supervivencia']['max_error']:.2f}",
             '✓' if self.metricas['yolo']['max_error'] < self.metricas['supervivencia']['max_error'] else ''],
            ['Error % ', f"{self.metricas['yolo']['error_pct']:.2f}%", 
             f"{self.metricas['supervivencia']['error_pct']:.2f}%",
             '✓' if self.metricas['yolo']['error_pct'] < self.metricas['supervivencia']['error_pct'] else ''],
            ['Desv. Std (cm)', f"{self.metricas['yolo']['std']:.2f}", 
             f"{self.metricas['supervivencia']['std']:.2f}",
             '✓' if self.metricas['yolo']['std'] < self.metricas['supervivencia']['std'] else ''],
            ['Correlación', f"{self.metricas['yolo']['correlacion']:.4f}", 
             f"{self.metricas['supervivencia']['correlacion']:.4f}",
             '✓' if self.metricas['yolo']['correlacion'] > self.metricas['supervivencia']['correlacion'] else ''],
            ['R²', f"{self.metricas['yolo']['r2']:.4f}", 
             f"{self.metricas['supervivencia']['r2']:.4f}",
             '✓' if self.metricas['yolo']['r2'] > self.metricas['supervivencia']['r2'] else ''],
        ]
        
        table = ax7.table(cellText=metricas_tabla, cellLoc='center', loc='center',
                         colWidths=[0.35, 0.2, 0.25, 0.1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Colorear header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Colorear filas alternadas
        for i in range(1, len(metricas_tabla)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        ax7.set_title('Resumen Comparativo de Métricas', fontsize=12, fontweight='bold', pad=20)
        
        # Título general
        fig.suptitle(f'Análisis Estadístico Completo - {self.session_path.name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Guardar
        output_file = self.analysis_path / 'analisis_estadistico_completo.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Gráficos guardados: {output_file}")
        
        plt.close()
    
    def guardar_resumen_json(self):
        """Guarda un resumen de las métricas en formato JSON."""
        if not self.metricas:
            return
        
        output_file = self.analysis_path / 'metricas_estadisticas.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.metricas, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Métricas JSON guardadas: {output_file}")
    
    def guardar_resumen_txt(self):
        """Guarda un resumen de las métricas en formato TXT legible."""
        if not self.metricas:
            return
        
        output_file = self.analysis_path / 'resumen_analisis.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RESUMEN DE ANÁLISIS ESTADÍSTICO - ODOMETRÍA VISUAL\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Sesión: {self.session_path.name}\n")
            f.write(f"Distancia de referencia: {self.metricas['comparacion']['distancia_referencia']:.1f} cm\n")
            f.write(f"Total de mediciones: {self.metricas['comparacion']['num_mediciones']}\n")
            f.write(f"Fecha de análisis: {pd.Timestamp.now()}\n\n")
            
            # YOLO
            f.write("-"*70 + "\n")
            f.write("YOLO - Métricas de Error\n")
            f.write("-"*70 + "\n")
            y = self.metricas['yolo']
            f.write(f"  MSE (Error Cuadrático Medio):     {y['mse']:8.2f} cm²\n")
            f.write(f"  RMSE (Raíz MSE):                  {y['rmse']:8.2f} cm\n")
            f.write(f"  MAE (Error Absoluto Medio):       {y['mae']:8.2f} cm\n")
            f.write(f"  Error Máximo:                     {y['max_error']:8.2f} cm\n")
            f.write(f"  Error Porcentual:                 {y['error_pct']:8.2f} %\n")
            f.write(f"  Desviación Estándar:              {y['std']:8.2f} cm\n")
            f.write(f"  Coeficiente de Variación:         {y['cv']:8.2f} %\n")
            f.write(f"  Error Acumulado Total:            {y['error_acumulado']:8.2f} cm\n")
            f.write(f"  Correlación con distancia real:   {y['correlacion']:8.4f} (p={y['pvalue_corr']:.4f})\n")
            f.write(f"  R² (Coef. determinación):         {y['r2']:8.4f}\n\n")
            
            # Supervivencia
            f.write("-"*70 + "\n")
            f.write("SUPERVIVENCIA - Métricas de Error\n")
            f.write("-"*70 + "\n")
            s = self.metricas['supervivencia']
            f.write(f"  MSE (Error Cuadrático Medio):     {s['mse']:8.2f} cm²\n")
            f.write(f"  RMSE (Raíz MSE):                  {s['rmse']:8.2f} cm\n")
            f.write(f"  MAE (Error Absoluto Medio):       {s['mae']:8.2f} cm\n")
            f.write(f"  Error Máximo:                     {s['max_error']:8.2f} cm\n")
            f.write(f"  Error Porcentual:                 {s['error_pct']:8.2f} %\n")
            f.write(f"  Desviación Estándar:              {s['std']:8.2f} cm\n")
            f.write(f"  Coeficiente de Variación:         {s['cv']:8.2f} %\n")
            f.write(f"  Error Acumulado Total:            {s['error_acumulado']:8.2f} cm\n")
            f.write(f"  Correlación con distancia real:   {s['correlacion']:8.4f} (p={s['pvalue_corr']:.4f})\n")
            f.write(f"  R² (Coef. determinación):         {s['r2']:8.4f}\n\n")
            
            # Comparación
            f.write("-"*70 + "\n")
            f.write("COMPARACIÓN ENTRE MÉTODOS\n")
            f.write("-"*70 + "\n")
            c = self.metricas['comparacion']
            f.write(f"  Correlación YOLO ↔ Supervivencia: {c['correlacion_entre_metodos']:8.4f} (p={c['pvalue_metodos']:.4f})\n\n")
            f.write(f"  Mejor algoritmo por segmento:\n")
            for metodo, conteo in c['mejor_conteo'].items():
                porcentaje = (conteo / c['num_mediciones']) * 100
                f.write(f"    {metodo:20s}: {conteo:3d} veces ({porcentaje:5.1f}%)\n")
            
            # Conclusión
            f.write("\n" + "-"*70 + "\n")
            f.write("CONCLUSIÓN\n")
            f.write("-"*70 + "\n")
            mejor_rmse = "YOLO" if y['rmse'] < s['rmse'] else "Supervivencia"
            mejor_mae = "YOLO" if y['mae'] < s['mae'] else "Supervivencia"
            mejor_corr = "YOLO" if y['correlacion'] > s['correlacion'] else "Supervivencia"
            
            f.write(f"  Menor RMSE:           {mejor_rmse}\n")
            f.write(f"  Menor MAE:            {mejor_mae}\n")
            f.write(f"  Mayor Correlación:    {mejor_corr}\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Resumen TXT guardado: {output_file}")
    
    def ejecutar_analisis_completo(self) -> bool:
        """
        Ejecuta el análisis estadístico completo.
        
        Returns:
            True si el análisis se completó con éxito, False en caso contrario
        """
        print("\n" + "🔬"*35)
        print("INICIANDO ANÁLISIS ESTADÍSTICO AUTOMÁTICO")
        print("🔬"*35)
        
        # Cargar datos
        if not self.cargar_datos():
            return False
        
        # Calcular métricas
        print("\n📊 Calculando métricas estadísticas...")
        self.calcular_metricas()
        
        # Imprimir en consola
        self.imprimir_metricas()
        
        # Generar gráficos
        print("\n📈 Generando gráficos estadísticos...")
        self.generar_graficos()
        
        # Guardar resúmenes
        print("\n💾 Guardando resúmenes...")
        self.guardar_resumen_json()
        self.guardar_resumen_txt()
        
        print("\n" + "✅"*35)
        print(f"ANÁLISIS COMPLETADO - Resultados en: {self.analysis_path}")
        print("✅"*35 + "\n")
        
        return True


def ejecutar_analisis_sesion(session_path: str) -> bool:
    """
    Función de conveniencia para ejecutar análisis desde otros módulos.
    
    Args:
        session_path: Ruta de la carpeta de sesión
        
    Returns:
        True si el análisis se completó con éxito
    """
    try:
        analizador = AnalizadorEstadistico(session_path)
        return analizador.ejecutar_analisis_completo()
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Para ejecutar manualmente
    import sys
    
    if len(sys.argv) > 1:
        session_path = sys.argv[1]
    else:
        # Buscar la sesión más reciente
        import glob
        sessions = sorted(glob.glob("report/sesion_*"))
        if sessions:
            session_path = sessions[-1]
        else:
            print("❌ No se encontraron sesiones en report/")
            sys.exit(1)
    
    ejecutar_analisis_sesion(session_path)
