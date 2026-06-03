# 🚀 Guía Rápida del Modelo AURA-ML2

---

## ✨ En una frase

**Un clasificador de razas de perros** que usa una red neuronal (EfficientNet) pre-entrenada en 1.2 millones de fotos, luego ajustada para reconocer 120 razas distintas.

---

## 🎯 ¿Cómo explicarlo fácilmente?

### **Analogía Simple**

Imagina que tienes un **experto en perros** que ha visto millones de fotos (ImageNet). Luego lo entrenas especializándolo solo en 120 razas (Stanford Dogs). Ahora puede identificar casi cualquier perro que le muestres.

**Eso es tu modelo.**

---

## 📊 Números Clave

| Concepto | Valor | 💡 Significado |
|----------|-------|---|
| **Modelo base** | EfficientNetB0 | Red neuronal pequeña pero poderosa |
| **Clases** | 120 razas | Reconoce 120 tipos de perros distintos |
| **Tamaño** | 18.8 MB | Cabe en un móvil sin problemas |
| **Precisión** | ~95% | Acierta 95 de cada 100 veces |
| **Velocidad** | 100-200ms | Una foto se analiza en menos de 1 segundo |

---

## 🏗️ ¿De qué está hecho?

### **El "cerebro" del modelo** (EfficientNet)
```
Capas de convolución (convolutions)
       ↓
Detectan patrones (orejas, ojos, cola, pelaje)
       ↓
Capas profundas combinan patrones
       ↓
Reconocen razas específicas
```

### **La "capacidad de decisión"** (Cabeza de clasificación)
```
Capa densa con 120 salidas
       ↓
Una para cada raza
       ↓
Devuelve probabilidad (0-100%) para cada una
       ↓
Ejemplo: "60% Labrador, 20% Golden Retriever, 15% ..."
```

---

## 💻 Cómo se Usa

### **Opción 1: A través de la App Flutter**

```
1. Abre Aura en tu teléfono
2. Toma foto de un perro
3. App envía la foto al servidor
4. Servidor ejecuta el modelo
5. ¡Recibes el resultado!
```

### **Opción 2: Desde el terminal (para pruebas)**

```bash
# Activar virtual environment
cd c:\Users\josel\OneDrive\Desktop\AURA-ML2
python -m venv auraml  # Si no existe
.\auraml\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar API
python -m uvicorn app.main:app --reload

# En otra terminal, probar:
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/dog_photo.jpg"
```

### **Opción 3: Python directo**

```python
from PIL import Image
from app.services.model import predict_topk

img = Image.open("mi_perro.jpg")
resultado = predict_topk(img, k=5)

print(f"Raza principal: {resultado['top1']['label']}")
print(f"Confianza: {resultado['top1']['score']:.2%}")
```

---

## 📈 ¿Qué devuelve el modelo?

### **Respuesta típica**

```json
{
  "top1": {
    "label": "n02105641-labrador_retriever",
    "score": 0.8923
  },
  "top5": [
    {"label": "labrador_retriever", "score": 0.8923},
    {"label": "golden_retriever", "score": 0.0754},
    {"label": "chesapeake_bay_retriever", "score": 0.0245},
    {"label": "curly-coated_retriever", "score": 0.0067},
    {"label": "english_springer_spaniel", "score": 0.0011}
  ],
  "log_id": 12345
}
```

**Interpretación**:
- ✅ **top1**: Raza más probable (89% de confianza)
- ✅ **top5**: Las 5 razas más probables
- ✅ **log_id**: ID guardado en base de datos para historial

---

## 🎓 ¿Cómo se Entrenó?

### **Paso 1: Recolectar datos**
```
Stanford Dogs Dataset (TFDS)
└── 120 razas de perros
    └── ~20,000 imágenes
```

### **Paso 2: Preparar el modelo**
```python
# Cargar EfficientNetB0 (pre-entrenado en ImageNet)
modelo = EfficientNetB0(pesos="imagenet")

# Congelar el backbone (mantener su conocimiento)
modelo.backbone.trainable = False

# Añadir cabeza personalizada (120 clases)
modelo.add(Dense(120, activation="softmax"))
```

### **Paso 3: Entrenar en dos fases**

**Fase 1** (12 épocas): Backbone congelado
```
- El modelo aprende a clasificar las 120 razas
- Sin destruir las características aprendidas de ImageNet
- Learning rate alto (1e-3)
```

**Fase 2** (1 época): Fine-tuning
```
- Descongelar últimas 60 capas
- Ajuste fino con learning rate muy bajo (1e-5)
- Mejora pequeña pero importante
```

### **Paso 4: Exportar**
```
model.keras ← Listo para producción
```

---

## 🔍 ¿Cuáles son sus Fuerzas y Debilidades?

### ✅ **Lo hace BIEN**
- ✓ Reconoce perros puros
- ✓ Funciona rápido (móvil-friendly)
- ✓ Muy preciso (~95% accuracy)
- ✓ Maneja diferentes ángulos
- ✓ Funciona con fotos de mala calidad

### ⚠️ **Lo hace MAL**
- ✗ Confunde razas parecidas
- ✗ No reconoce gatos, pájaros, etc.
- ✗ Falla si la foto es muy borrosa
- ✗ Ángulos muy raros pueden confundirlo
- ✗ No funciona con dibujos

---

## 🛠️ Mejoras Futuras

### **Corto plazo**
- [ ] Agregar más razas (150+)
- [ ] Mejorar con fine-tuning más intenso
- [ ] Crear reportes por raza

### **Mediano plazo**
- [ ] Pasar a EfficientNetB1 o B2 (más preciso)
- [ ] Agregar detección de múltiples perros
- [ ] Detección de razas cruzadas

### **Largo plazo**
- [ ] Multiclase (perros, gatos, pájaros)
- [ ] Edge AI (ejecutar en el teléfono sin servidor)
- [ ] Explicabilidad (mostrar qué partes detectó)

---

## 📝 Parámetros Técnicos (para devs)

```yaml
# Architecture
backbone: EfficientNetB0
input_shape: (224, 224, 3)
num_classes: 120
total_params: 5.3M (backbone) + 0.24M (head)

# Training
dataset: stanford_dogs (TFDS)
batch_size: 32
epochs_phase1: 12
epochs_phase2: 1
optimizer: Adam
loss: sparse_categorical_crossentropy

# Export
format: .keras (recomendado)
size: 18.8 MB
also_exported: SavedModel/ (TF Serving compatible)

# Performance
test_accuracy: ~94-95%
inference_time: 100-200ms (GPU), 500ms (CPU)
throughput: 5-10 img/s
memory_footprint: ~100 MB (en RAM)
```

---

## 🎬 Ejemplo de Uso Práctico

### **Escenario: Usuario toma foto de su perro en Aura**

```
1. Usuario: "Abre cámara"
2. Aura: Muestra preview en tiempo real
3. Usuario: Toma foto
4. Aura: Comprime y envía a servidor
   └─> POST /predict con imagen

5. Servidor:
   └─> Carga imagen
   └─> Redimensiona a 224×224
   └─> Normaliza ([-1, 1])
   └─> Ejecuta modelo.predict()
   └─> Obtiene 120 probabilidades
   └─> Selecciona top-5
   └─> Guarda en DB (log_id=123)
   └─> Devuelve JSON

6. Aura recibe respuesta:
   {
     "top1": {"label": "labrador", "score": 0.89},
     "top5": [...],
     "log_id": 123
   }

7. Aura muestra: "¡Es un Labrador! (89% de confianza)"
8. Usuario puede guardar en historial
```

---

## 🚀 Para Presentar el Proyecto

### **Elevator Pitch (30 segundos)**

*"AURA es una app que identifica razas de perros usando Inteligencia Artificial. Tomas una foto y te dice exactamente qué raza es con 95% de precisión. Usamos EfficientNet, un modelo entrenado en 20,000 imágenes de 120 razas distintas."*

### **Presentación Técnica (5 minutos)**

1. **Problema**: Usuarios quieren saber qué raza es su perro
2. **Solución**: App con ML integrado
3. **Tecnología**: 
   - EfficientNetB0 (modelo base)
   - Transfer Learning (eficiente)
   - FastAPI (backend)
   - Flutter (frontend móvil)
4. **Resultados**: 95% accuracy, 200ms por predicción
5. **Deploy**: En servidor PostgreSQL + API REST

### **Demostración Live**

```bash
# Terminal 1: Iniciar API
python -m uvicorn app.main:app --reload

# Terminal 2: Test con curl
curl -X POST http://localhost:8000/predict \
  -F "file=@labrador.jpg"

# Mostrar el JSON de respuesta en la pantalla
```

---

## 📚 Referencias

- **Dataset**: [Stanford Dogs (TFDS)](https://www.tensorflow.org/datasets/catalog/stanford_dogs)
- **Modelo**: [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- **Framework**: TensorFlow 2.16+
- **Archivo Principal**: `entrenamiento.py` (cómo se creó)
- **Archivo Predicción**: `app/services/model.py` (cómo se usa)

---

**Creado**: 5 de Mayo, 2026  
**Modelo Entrenado**: 11 de Noviembre, 2025  
**Versión**: 1.0
