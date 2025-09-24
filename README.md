# Tennis Match Predictor (ATP)


![Status](https://img.shields.io/badge/status-in%20development-yellow)

Este proyecto implementa un pipeline de Machine Learning para predecir el resultado de partidos de tenis ATP a partir de datos históricos de jugadores, torneos y estadísticas de rendimiento.  

Actualmente se encuentra en **desarrollo activo**.  No se han subido todos los datos necesarios al repositorio para poder ejecutar el pipeline de principio a fin. Los datos utilizados provienen en gran medida del repositorio [tennis_atp de Jeff Sackmann](https://github.com/JeffSackmann/tennis_atp), ampliamente reconocido como fuente de resultados históricos de tenis profesional.  

## Estructura del proyecto

  ```bash
  ├── src/ # Código fuente principal
  │ ├── pipeline/ # Scripts del pipeline de datos y entrenamiento
  │ ├── predict.py # Predicciones con el modelo entrenado
  │ └── utils.py # Funciones auxiliares
  ├── scripts/ # Scripts de ejecución (ej. run_all.py)
  ├── experiments/ # Experimentos y tuner de hiperparámetros (Optuna)
  ├── artifacts/ # Artefactos generados (datasets, modelos, reportes)
  ├── data/ # Datos crudos (no incluidos en el repo)
  ├── requirements.txt # Dependencias del proyecto
  └── README.md

  ```

Carpetas como `data/`, `artifacts/`, `venv/` y `__pycache__/` se encuentran ignoradas en Git porque contienen datos pesados, entornos o archivos temporales.

## Cómo funciona el pipeline

El código está diseñado como una **secuencia de etapas** que enriquecen progresivamente los datos hasta producir un dataset listo para entrenar un modelo predictivo:

1. **Enriquecimiento de datos básicos**  
   `enriquecer_datos.py` unifica partidos de distintas temporadas, corrige identificadores, genera IDs únicos en caso de ausencia, y calcula métricas iniciales como Elo global y por superficie, winrates previos y balance head-to-head entre jugadores.

2. **Forma y descanso de jugadores**  
   `enriquecer_datos_forma_y_descanso.py` añade estadísticas de forma reciente (últimos 5, 10 y 25 partidos), días de descanso desde el último encuentro y estimaciones de fatiga a partir de la duración media de los partidos previos.

3. **Forma por superficie**  
   `enriquecer_forma_superficie.py` incorpora winrates específicos de cada jugador en cada superficie (hard, clay, grass, carpet) también en ventanas temporales cortas.

4. **Preparación del dataset**  
   `preparar_datos.py` transforma los datos enriquecidos en un formato orientado a aprendizaje supervisado.  
   Cada partido se representa dos veces: una tomando al ganador como jugador A (con etiqueta `y=1`) y otra tomando al perdedor como jugador A (con etiqueta `y=0`).  
   Se generan variables diferenciales (`A - B`) como diferencias de ranking, puntos, Elo, forma reciente, fatiga, entre otras.

5. **Entrenamiento del modelo**  
   `entrenar_modelo.py` entrena un clasificador basado en **XGBoost**.  
   Se usa validación temporal:  
   - Entrenamiento con datos previos a 2024.  
   - Validación con datos de 2024 (para calibrar umbral y ajustar probabilidades).  
   - Test con datos de 2025 en adelante.  

   El modelo se calibra opcionalmente con regresión isotónica y se evalúa con métricas como LogLoss, Brier Score, AUC y Accuracy.

6. **Predicción**  
   `predict.py` carga un modelo previamente entrenado y permite generar predicciones sobre nuevos partidos.


## Metodología teórica

El proyecto combina distintas ideas de modelado en deportes:

- **Elo adaptado al tenis**  
  Cada jugador tiene un rating global y uno específico por superficie, con decaimiento temporal y ajustes por ronda y formato (BO3/BO5). Esto permite capturar la dinámica de forma y especialización.

- **Rolling winrates**  
  En lugar de usar estadísticas acumuladas, se consideran ventanas de los últimos N partidos (5, 10, 25), que capturan la forma reciente sin sesgo por carreras largas.

- **Fatiga y descanso**  
  El tiempo medio en cancha y los días desde el último partido se incluyen como variables que afectan el rendimiento inmediato.

- **Codificación diferencial de features**  
  En lugar de usar valores absolutos, se generan diferencias jugador A – jugador B. Esto es estándar en modelos de enfrentamientos, porque la probabilidad de victoria depende de la comparación relativa entre los jugadores.

- **Validación temporal estricta**  
  Se entrena en el pasado y se valida/testea en intervalos posteriores, simulando cómo se usaría el modelo en producción (predicción de partidos futuros).

- **Calibración de probabilidades**  
  Tras entrenar, se ajustan las probabilidades para que sean bien calibradas, de forma que un 0.7 predicho represente aproximadamente un 70% de victorias reales.


## Instalación y uso

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/<TU_USUARIO>/tennis-match-predictor.git
   cd tennis-match-predictor

2. Crear un entorno virtual e instalar dependencias:
   ```bash
    python -m venv venv
    source venv/bin/activate    # Linux/Mac
    venv\Scripts\activate       # Windows

3. Ejecutar el pipeline completo:
   ```bash
    pip install -r requirements.txt
    python scripts/run_all.py


## Estado actual

El pipeline está implementado y genera datasets listos para entrenar modelos.

Se ha integrado un modelo base con XGBoost y tuner de hiperparámetros con Optuna.

La precisión actual ronda el 75% en test de 2025, con AUC ~0.80.

El proyecto está en fase de desarrollo: se planean nuevas features (estadísticas de return, fuerza de rivales, viajes entre continentes, etc.) y mejoras en calibración y umbrales.

## Roadmap

- Implementar API para predicciones en tiempo real.

- Explorar modelos alternativos (LightGBM, CatBoost).

## Autor
[Liam Saborido Sueiro]([https://github.com/JeffSackmann/tennis_atp](https://www.linkedin.com/in/liam-saborido-b245352b5/))


