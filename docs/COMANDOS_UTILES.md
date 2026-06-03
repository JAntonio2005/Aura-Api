# 🛠️ Comandos Útiles - AURA-ML2

---

## ⚙️ **Setup Inicial**

### **Crear Virtual Environment (Recomendado)**

```bash
# En Windows (PowerShell)
cd c:\Users\josel\OneDrive\Desktop\AURA-ML2
python -m venv auraml

# Activar
.\auraml\Scripts\activate
```

### **Instalar Dependencias**

```bash
# Instalar todo
pip install -r requirements.txt

# O instalar solo ML
pip install tensorflow>=2.16,<2.18 tensorflow-datasets numpy pillow

# O instalar solo API
pip install fastapi uvicorn python-multipart
```

---

## 🚀 **Ejecutar el Modelo**

### **Iniciar API FastAPI**

```bash
# Modo básico
python -m uvicorn app.main:app --reload

# Con puerto específico
python -m uvicorn app.main:app --reload --port 8000

# Producción (sin reload)
python -m uvicorn app.main:app --workers 4 --host 0.0.0.0
```

**Acceso**: `http://localhost:8000`  
**Docs**: `http://localhost:8000/docs` (Swagger UI)

---

### **Entrenar Modelo**

```bash
# Default (12 épocas, no fine-tune)
python entrenamiento.py

# Con fine-tuning
python entrenamiento.py --epochs 12 --fine_tune 3

# Customizado
python entrenamiento.py \
  --epochs 20 \
  --fine_tune 2 \
  --batch_size 32 \
  --img_size 224 \
  --learning_rate 0.001 \
  --ft_learning_rate 0.00001 \
  --output_dir ./exports

# Reanudar desde checkpoint
python entrenamiento.py --resume_from auto --fine_tune 2

# Con mixed precision (GPU)
python entrenamiento.py --mixed_precision --cache
```

---

## 🧪 **Probar el Modelo**

### **Via cURL (Terminal)**

```bash
# Predicción simple
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/dog.jpg"

# Con enrich (añade información de raza)
curl -X POST "http://localhost:8000/predict?enrich=true" \
  -F "file=@path/to/dog.jpg" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### **Con Python**

```python
from PIL import Image
from app.services.model import predict_topk, display_name

# Cargar imagen
img = Image.open("mi_perro.jpg")

# Predicción
resultado = predict_topk(img, k=5)

# Imprimir
print(f"Raza: {display_name(resultado['top1']['label'])}")
print(f"Confianza: {resultado['top1']['score']:.2%}")
for i, pred in enumerate(resultado['top5'], 1):
    print(f"{i}. {display_name(pred['label'])}: {pred['score']:.2%}")
```

### **Con Requests (HTTP)**

```python
import requests

# Preparar archivo
with open("dog.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/predict",
        files=files
    )

# Resultado
resultado = response.json()
print(resultado)
```

---

## 📊 **Inspeccionar Modelo**

### **Ver estructura del modelo**

```python
import tensorflow as tf

# Cargar modelo
modelo = tf.keras.models.load_model(
    "exports/stanford-dogs_20251111-211940/model.keras"
)

# Resumen
modelo.summary()

# Info de capas
for layer in modelo.layers:
    print(f"{layer.name}: {layer.output_shape}")

# Parámetros totales
print(f"Total params: {modelo.count_params():,}")
```

### **Ver labels disponibles**

```python
import json

labels = json.load(
    open("exports/stanford-dogs_20251111-211940/labels.json")
)

print(f"Total razas: {len(labels)}")
print(f"Primeras 10: {labels[:10]}")
```

### **Tamaño del modelo**

```bash
# En terminal (Windows)
dir "exports/stanford-dogs_20251111-211940/model.keras"

# En terminal (Linux/Mac)
ls -lh exports/stanford-dogs_20251111-211940/model.keras
```

---

## 🔍 **Debug y Diagnóstico**

### **Ver logs de entrenamiento**

```python
# En el código de entrenamiento
import tensorflow as tf

# Habilitar logs verbosos
tf.debugging.set_log_device_placement(True)

# Ver versión TF
print(tf.__version__)

# Ver si GPU está disponible
print(tf.config.list_physical_devices('GPU'))
```

### **Verificar predicciones**

```python
import numpy as np
from PIL import Image
import tensorflow as tf
from app.services.model import preprocess_pil, model, CLASS_NAMES

# Cargar imagen
img = Image.open("dog.jpg")

# Preprocesar
x = preprocess_pil(img)
print(f"Input shape: {x.shape}")

# Predecir
predictions = model.predict(x)
print(f"Output shape: {predictions.shape}")

# Top-5
top_indices = np.argsort(-predictions[0])[:5]
for i, idx in enumerate(top_indices, 1):
    score = predictions[0][idx]
    label = CLASS_NAMES[idx]
    print(f"{i}. {label}: {score:.4f}")
```

---

## 📈 **Analizar Rendimiento**

### **Metrics en Test**

```bash
# Entrenar y evaluar
python entrenamiento.py --epochs 10 --fine_tune 1

# Verá el test_accuracy al final
# Ejemplo: [Eval] Test accuracy: 0.9423
```

### **Generar Matriz de Confusión**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# Cargar datos y modelo
modelo = tf.keras.models.load_model("model.keras")
# ... cargar test_ds y CLASS_NAMES

# Predicciones
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = modelo.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Report de clasificación
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Visualizar (opcional)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.show()
```

---

## 🧹 **Mantenimiento**

### **Limpiar caché TensorFlow**

```bash
# Windows
rmdir /s %USERPROFILE%\.cache\tensorflow

# Linux/Mac
rm -rf ~/.cache/tensorflow
```

### **Limpiar archivos temporales**

```bash
# Remover carpeta de exports viejos (cautela)
cd exports
dir /b /o-d  # Ver ordenado por fecha
# Eliminar manualmente carpetas viejas
```

### **Actualizar TensorFlow**

```bash
# Ver versión actual
pip show tensorflow

# Actualizar
pip install --upgrade tensorflow

# Verificar compatibilidad
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 🔐 **Seguridad & Auth**

### **Obtener Token JWT**

```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password123"}'

# Respuesta:
# {"access_token":"eyJ..."}
```

### **Usar Token en Predicción**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer eyJ..." \
  -F "file=@dog.jpg"
```

---

## 📊 **Ver Historial de Predicciones**

### **Acceder a BD**

```bash
# Conectar a PostgreSQL
psql -U usuario -d aura_db

# Ver predicciones
SELECT * FROM predictionlog ORDER BY created_at DESC LIMIT 10;

# Estadísticas
SELECT top1_label, COUNT(*) as count 
FROM predictionlog 
GROUP BY top1_label 
ORDER BY count DESC;
```

---

## 🐳 **Docker (Opcional)**

### **Build imagen**

```bash
# Ver Dockerfile existente
cat Dockerfile

# Build
docker build -t aura-ml2 .

# Run
docker run -p 8000:8000 aura-ml2
```

---

## 📝 **Troubleshooting Común**

### **Error: "No module named tensorflow"**

```bash
pip install tensorflow>=2.16,<2.18
```

### **Error: "CUDA out of memory"**

```python
# En código:
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
)
```

### **Error: "File not found: model.keras"**

```bash
# Verificar que el archivo existe
dir "exports/stanford-dogs_20251111-211940/model.keras"

# Si no existe, entrenar primero
python entrenamiento.py
```

### **Error: "Connection refused" en API**

```bash
# Verificar que FastAPI está corriendo
python -m uvicorn app.main:app --reload

# Verificar puerto
curl http://localhost:8000/docs
```

---

## 🎯 **Quick Checklist**

- [ ] ✅ Virtual env activado
- [ ] ✅ Dependencies instaladas (`pip install -r requirements.txt`)
- [ ] ✅ API corriendo (`uvicorn app.main:app --reload`)
- [ ] ✅ Modelo descargado (`exports/stanford-dogs_.../model.keras`)
- [ ] ✅ Acceso a `http://localhost:8000/docs`
- [ ] ✅ Predicción funciona (`curl /predict`)
- [ ] ✅ Base de datos conectada
- [ ] ✅ JWT auth funcionando

---

## 📚 **Recursos Útiles**

- **TensorFlow Docs**: https://tensorflow.org/
- **Keras**: https://keras.io/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Stanford Dogs**: https://www.tensorflow.org/datasets/catalog/stanford_dogs
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946

---

**Last Updated**: 5 de Mayo, 2026
