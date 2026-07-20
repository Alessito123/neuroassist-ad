# NeuroAssist AD

Aplicación web modular para análisis y diagnóstico **asistido y educativo** de Alzheimer a partir
de biomarcadores tabulares. Incluye carga pública automática, PostgreSQL/Neon, EDA, cinco modelos,
validación estadística, optimización, inferencia individual y reportes PDF.

> **Aviso médico y de datos:** no es un dispositivo médico ni sustituye el diagnóstico profesional.
> El dataset predeterminado es sintético y educativo; un resultado no puede extrapolarse a pacientes
> reales sin validación externa, calibración, gobernanza y aprobación regulatoria aplicable.

## Funcionalidades

- CSV, Excel y NIfTI (`.nii`/`.nii.gz`); para NIfTI se extraen siete descriptores volumétricos básicos.
- Descarga automática del [Alzheimer's Disease Dataset de Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset),
  con 2,149 registros y 35 variables.
- Imputación, one-hot encoding, escalado estándar/robusto y SMOTE/ADASYN dentro de cada fold.
- Random Forest, XGBoost y SVM RBF; Stacking con regresión logística y Voting suave ponderado.
- StratifiedKFold de 3, 5 o 10 particiones; Accuracy, Precision, Recall, F1, AUC-ROC y AUC-PR.
- IC 95 % por folds, McNemar pareado sobre predicciones out-of-fold y Friedman global.
- RandomizedSearchCV y persistencia de parámetros/modelo serializado en PostgreSQL.
- PDF con resumen, métricas, curvas, matriz de confusión, importancia y resultado individual.

## Inicio rápido

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Abra `http://localhost:8501`. Si no define `DATABASE_URL`, la aplicación usa SQLite local únicamente
para desarrollo.

Con Docker y PostgreSQL local:

```powershell
docker compose up --build
```

## Neon

1. Cree un proyecto y copie la URL PostgreSQL con `sslmode=require`.
2. Defina `DATABASE_URL` en el entorno donde corre Streamlit.
3. Pulse **Verificar conexión y crear tablas**. SQLAlchemy crea `datasets_raw`,
   `datasets_processed`, `modelos_entrenados` y `resultados_diagnostico`.

Las credenciales no se imprimen ni se guardan en Git. Use `.env.example` como referencia.

## Despliegue y Vercel

Streamlit necesita un proceso Python persistente y su propio protocolo de sesión sobre WebSocket.
Vercel documenta funciones Python ASGI/WSGI y, desde junio de 2026, WebSockets en beta mediante el
runtime Node.js; no documenta Streamlit como entrypoint Python soportado. Por eso este repositorio
separa:

1. **Aplicación real:** contenedor `Dockerfile` ejecutado en un host persistente compatible con
   Streamlit (o Streamlit Community Cloud).
2. **Gateway Vercel:** `api/index.js` redirige a la aplicación real mediante `STREAMLIT_APP_URL` y
   expone `/health`.

Despliegue el gateway:

```powershell
npx vercel
npx vercel env add STREAMLIT_APP_URL production
npx vercel --prod
```

Esta separación evita afirmar que un landing estático ejecuta el motor ML. Si Vercel incorpora un
runtime Python persistente compatible con Streamlit, puede sustituirse el gateway sin cambiar los
módulos de dominio.

Referencias: [runtime Python de Vercel](https://vercel.com/docs/functions/runtimes/python) y
[WebSockets en Vercel](https://vercel.com/changelog/websocket-support-is-now-in-public-beta).

## Arquitectura

```text
app.py                 interfaz y orquestación Streamlit
config.py              entorno, semillas y fuente pública
database.py            ORM, conexión y persistencia
preprocessing.py       carga, NIfTI y pipeline sin leakage
models.py              modelos, CV, métricas, IC y tests estadísticos
visualization.py       EDA, PCA, ROC/PR e interpretación
report_generator.py    PDF en memoria
api/                   gateway liviano para Vercel
tests/                 pruebas funcionales esenciales
```

## Lógica de los modelos híbridos

**Stacking** genera probabilidades out-of-fold de RF, XGBoost y SVM y entrena una regresión logística
como meta-modelo. Así puede aprender qué modelo es más fiable en distintas regiones sin usar como
entrada las predicciones in-sample del mismo caso.

**Voting suave** promedia probabilidades ponderadas (RF y XGBoost reciben peso 2; SVM peso 1). Es una
combinación más estable y transparente: un error extremo de un modelo puede moderarse con los demás.

En un problema médico se deben leer conjuntamente sensibilidad/falsos negativos, AUC-PR, calibración,
matriz de confusión e intervalos, no solo accuracy. El ganador usa el promedio AUC-ROC + F1 como regla
operativa y se acompaña de Friedman/McNemar; esto no reemplaza validación prospectiva externa.

## Pruebas

```powershell
pytest -q
ruff check .
```

La semilla global es 42 para que particiones, sobremuestreo y modelos sean reproducibles.

