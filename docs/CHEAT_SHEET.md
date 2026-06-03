# 🐕 Cheat Sheet: AURA-ML2 en 1 Minuto

---

## 🎯 **En una sola frase**

Un modelo de IA que mira fotos de perros y dice qué raza son con 95% de precisión.

---

## 📊 **3 Números Clave**

| # | Número | Significa |
|-|--------|----------|
| 1 | **120 razas** | Cantidad de razas que reconoce |
| 2 | **18.8 MB** | Tamaño del modelo (súper pequeño) |
| 3 | **0.2 segundos** | Tiempo para analizar una foto |

---

## 🏗️ **Arquitectura (Simplificada)**

```
FOTO → REDIMENSIONA → NORMALIZA → MODELO IA → TOP-5 RAZAS
```

---

## 🧠 **¿Qué es EfficientNetB0?**

Una red neuronal que:
1. ✅ Es pequeña (18.8 MB)
2. ✅ Es precisa (95% accuracy)
3. ✅ Es rápida (0.2 seg)
4. ✅ Está pre-entrenada (ImageNet)

---

## 📚 **Dataset**

- **Nombre**: Stanford Dogs
- **Fotos**: 20,000 imágenes de perros
- **Razas**: 120 clases
- **Fuente**: Dataset público de TensorFlow

---

## 🎓 **Proceso de Entrenamiento**

```
Fase 1 (12 épocas):
┌─────────────────────┐
│ Backbone congelado  │ ← Pesos de ImageNet se mantienen
│ Cabeza se entrena   │ ← Se aprende a clasificar 120 razas
└─────────────────────┘
         ↓
Fase 2 (1 época):
┌─────────────────────┐
│ Fine-tuning         │ ← Ajuste fino con learning rate bajo
│ Mejora precisión    │ ← 95% accuracy final
└─────────────────────┘
         ↓
Exportar: model.keras (18.8 MB)
```

---

## 💻 **Cómo Usarlo**

### **Opción 1: App Aura (Usuario)**
```
Toma foto → App envía → Servidor predice → ¡Resultado!
```

### **Opción 2: Terminal (Dev)**
```bash
python -m uvicorn app.main:app --reload
curl -X POST http://localhost:8000/predict -F "file=@dog.jpg"
```

### **Opción 3: Python (Dev)**
```python
from app.services.model import predict_topk
resultado = predict_topk(imagen, k=5)
```

---

## 📤 **Output (Respuesta)**

```json
{
  "top1": {"label": "labrador", "score": 0.89},
  "top5": [
    {"label": "labrador", "score": 0.89},
    {"label": "golden_retriever", "score": 0.07},
    {"label": "chesapeake_bay", "score": 0.02},
    {"label": "curly_coated", "score": 0.01},
    {"label": "english_springer", "score": 0.001}
  ],
  "log_id": 12345
}
```

---

## ✅ **Lo que Funciona Bien**

- ✅ Reconoce 120 razas
- ✅ Rápido (móvil-friendly)
- ✅ Preciso (95%)
- ✅ Pequeño (cabe en cualquier lado)
- ✅ Produce top-5 alternativas

---

## ❌ **Lo que NO Funciona**

- ❌ Otros animales (solo perros)
- ❌ Dibujos (solo fotos)
- ❌ Fotos muy borrosas
- ❌ Ángulos muy raros
- ❌ Razas cruzadas no claras

---

## 🔧 **Especificaciones Técnicas**

```yaml
Framework:           TensorFlow 2.16+
Modelo:              EfficientNetB0
Input:               (224, 224, 3)
Clases:              120
Parámetros totales:  5.5M
Tamaño archivo:      18.8 MB
Formato exportado:   .keras + SavedModel
Precisión test:      94-95%
Tiempo inferencia:   100-200ms (GPU)
                     500ms (CPU)
```

---

## 📁 **Archivos Clave**

```
exports/stanford-dogs_20251111-211940/
├── model.keras          ← EL MODELO (18.8 MB)
├── labels.json          ← 120 razas
└── saved_model/         ← Formato TF Serving

app/
├── services/model.py    ← Predicción
├── routers/predict.py   ← API endpoint
└── main.py              ← FastAPI app
```

---

## 🚀 **Flujo Producción**

```
1. Usuario toma foto (app Flutter)
2. Envía a servidor: POST /predict
3. Servidor carga imagen
4. Redimensiona a 224×224
5. Normaliza valores
6. Ejecuta modelo.predict()
7. Obtiene 120 probabilidades
8. Selecciona top-5
9. Guarda en BD (log_id)
10. Devuelve JSON
11. App muestra resultado
```

---

## 💡 **Analogía Rápida**

Piensa en un **experto en perros**:
- Ha visto 1.2M fotos de todo (experto general)
- Lo especializas en 120 razas (experto específico)
- Ahora identifica cualquier perro con 95% accuracy

**Eso es tu modelo.**

---

## 🎬 **Ejemplo en Acción**

**Input**: Foto de un Labrador  
**Output**:
```
🐕 Labrador Retriever (89% confianza)

Alternativas:
1. Golden Retriever (7%)
2. Chesapeake Bay (2%)
3. Curly-coated (1%)
4. English Springer (0.3%)
5. Duck Tolling Retriever (0.1%)

ID guardado: #12345
```

---

## 📈 **Comparación Rápida**

| Aspecto | Tu Modelo |
|---------|-----------|
| Precisión | 95% ✅ |
| Velocidad | 0.2s ✅ |
| Tamaño | 18.8 MB ✅ |
| Móvil-friendly | Sí ✅ |
| Listo producción | Sí ✅ |

---

## 🎓 **Para Diferentes Audiencias**

### **Tu abuela**: 
*"Es como un especialista en perros que mira fotos y dice la raza"*

### **Gerente no-tech**: 
*"Un IA que identifica razas con 95% precisión, tarda 0.2s por foto"*

### **Técnico**: 
*"EfficientNetB0 transfer learning, Stanford Dogs, 120 clases, 18.8MB"*

### **Inversor**: 
*"MVP listo, escalable, core de la app, diferenciador competitivo"*

---

## ✨ **Por Qué Es Bueno**

1. **Preciso**: 95% de acierto
2. **Rápido**: 0.2 segundos
3. **Pequeño**: Cabe en teléfono
4. **Probado**: Dataset público
5. **Production-ready**: Ya en uso

---

## 🔮 **Mejoras Posibles**

1. EfficientNetB1/B2 (más preciso)
2. Más razas (150+)
3. Detección múltiples perros
4. Reconocer razas cruzadas
5. Ejecutar en teléfono (offline)

---

## 📞 **Preguntas Frecuentes Rápidas**

**¿Funciona en internet lento?**  
Sí, solo envía la imagen (comprimida)

**¿Se calienta el teléfono?**  
No, la computación es en el servidor

**¿Funciona con fotos de baja calidad?**  
Sí, bastante bien

**¿Qué pasa si le muestro un gato?**  
Intentará clasificarlo como raza de perro (resultado erróneo)

**¿Se puede mejorar?**  
Sí, reentrenando con más datos

---

## 🎯 **Bottom Line**

Tu modelo es:
- 🎯 Específico (perros)
- 📊 Preciso (95%)
- ⚡ Rápido (0.2s)
- 💾 Pequeño (18.8 MB)
- ✅ Listo para producción

**Perfect fit para Aura.**

---

**Última actualización**: 5 de Mayo, 2026  
**Versión del modelo**: Entrenado 11/11/2025  
**Status**: 🟢 Production Ready
