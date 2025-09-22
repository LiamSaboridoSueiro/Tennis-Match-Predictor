# run_all.py (versión corta corregida)
import subprocess, os, sys

pipeline = [
    "src/pipeline/enriquecer_datos.py",
    "src/pipeline/enriquecer_datos_forma_y_descanso.py",
    "src/pipeline/enriquecer_forma_superficie.py",
    "src/pipeline/preparar_datos.py",
]


for script in pipeline:
    print("="*60)
    print(f"▶ Ejecutando {script}")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Script no encontrado: {script}")
    subprocess.run([sys.executable, script], check=True)  # usa el Python del venv

print("\n✅ Pipeline completo. Artefactos generados:")
print("  - partidos_enriquecidos.csv")
print("  - partidos_enriquecidos_v2.csv")
print("  - partidos_enriquecidos_final.csv")
print("  - dataset_modelo.csv")
