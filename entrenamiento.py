#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento de clasificador de razas (Stanford Dogs) con Keras/TensorFlow.
- Descarga automática via TFDS
- Preprocesamiento + data augmentation
- Entrenamiento (backbone congelado) + fine-tuning opcional (parcial por defecto)
- Resume opcional desde checkpoint
- Exporta: model.keras + SavedModel/ + labels.json + best.keras

Ejemplos:
  python entrenamiento.py --epochs 8 --fine_tune 4 --output_dir ./exports
  python entrenamiento.py --img_size 224 --batch_size 32 --mixed_precision
  python entrenamiento.py --fine_tune 2 --ft_last 60 --resume_from auto
"""

import json
import time
import argparse
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# -----------------------------
# Config por defecto
# -----------------------------
DEFAULT_IMG_SIZE = 224
DEFAULT_BATCH = 32
DEFAULT_EPOCHS = 12        # epochs fase congelada
DEFAULT_FINE_TUNE = 1      # epochs de fine-tuning
DEFAULT_FT_LAST = 60       # nº de capas a descongelar (0 = full FT)
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena un clasificador de razas (Stanford Dogs).")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE, help="Tamaño del lado de la imagen.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Épocas con el backbone congelado.")
    parser.add_argument("--fine_tune", type=int, default=DEFAULT_FINE_TUNE, help="Épocas de fine-tuning (0 = sin FT).")
    parser.add_argument("--ft_last", type=int, default=DEFAULT_FT_LAST,
                        help="Descongelar solo las últimas N capas en FT (0 = full FT).")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="LR inicial (fase congelada).")
    parser.add_argument("--ft_learning_rate", type=float, default=1e-5, help="LR para fine-tuning.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fracción del train para validación [0,1).")
    parser.add_argument("--output_dir", type=str, default="./exports", help="Directorio de salida.")
    parser.add_argument("--resume_from", type=str, default="",
                        help='Ruta a .keras para reanudar o "auto" para usar el último best.keras en output_dir.')
    parser.add_argument("--cache", action="store_true", help="Activar dataset.cache() (si tienes RAM suficiente).")
    parser.add_argument("--mixed_precision", action="store_true", help="Activar mixed precision (GPU/WSL).")
    return parser.parse_args()


def maybe_enable_mixed_precision(enabled: bool):
    if enabled:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[MP] Mixed precision activado (float16).")
        except Exception as e:
            print(f("[MP] No se pudo activar mixed precision: {e}"))


def load_dataset(img_size: int, batch_size: int, val_split: float, use_cache: bool = False):
    """Carga 'stanford_dogs' desde TFDS y crea splits train/val/test."""
    assert 0.0 <= val_split < 1.0, "val_split debe estar en [0,1)."
    print("[TFDS] Descargando/cargando 'stanford_dogs'…")

    ds_splits = tfds.load(
        "stanford_dogs",
        split=[f"train[:{int((1.0 - val_split)*100)}%]",
               f"train[{int((1.0 - val_split)*100)}%:]",  # val
               "test"],
        with_info=True,
        as_supervised=True,
        shuffle_files=True
    )
    (ds_train, ds_val, ds_test), ds_info = ds_splits

    num_classes = ds_info.features["label"].num_classes
    int2str = ds_info.features["label"].int2str
    class_names = [int2str(i) for i in range(num_classes)]

    # Preprocess y augment
    def preprocess(img, label):
        img = tf.image.resize(img, (img_size, img_size))
        img = tf.keras.applications.efficientnet.preprocess_input(img)  # [-1,1]
        return img, label

    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    def add_augment(img, label):
        return augment(img, training=True), label

    def pipeline(ds, training=False):
        if training:
            ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.map(add_augment, num_parallel_calls=AUTOTUNE)
        if use_cache:
            ds = ds.cache()
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds

    train_ds = pipeline(ds_train, training=True)
    val_ds = pipeline(ds_val, training=False)
    test_ds = pipeline(ds_test, training=False)

    return train_ds, val_ds, test_ds, class_names


def build_model(num_classes: int, img_size: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Modelo EfficientNetB0 + cabeza de clasificación; entrada flexible (None,None,3)."""
    inputs = tf.keras.Input(shape=(None, None, 3), dtype=tf.float32, name="image")
    x = layers.Resizing(img_size, img_size, name="resize")(inputs)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_tensor=x,
        weights="imagenet"
    )
    base.trainable = False  # fase 1 congelada

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dogbreed_efficientnetb0")
    return model, base


def compile_model(model: tf.keras.Model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def _find_backbone(m: tf.keras.Model):
    """Encuentra el submodelo EfficientNet dentro de un modelo cargado."""
    try:
        return m.get_layer("efficientnetb0")
    except Exception:
        for l in m.layers:
            if isinstance(l, tf.keras.Model) and "efficientnet" in l.name:
                return l
        raise RuntimeError("No se encontró el backbone EfficientNet en el modelo cargado.")


def _resolve_resume_path(output_dir: Path) -> Path | None:
    """Devuelve el último best.keras dentro de output_dir si existe."""
    all_ckpts = sorted(output_dir.glob("stanford-dogs_*/checkpoints/best.keras"))
    return all_ckpts[-1] if all_ckpts else None


def train_and_export(args):
    # Mixed precision (solo si tienes GPU/WSL; en CPU no aporta)
    maybe_enable_mixed_precision(args.mixed_precision)

    # Data
    train_ds, val_ds, test_ds, class_names = load_dataset(
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        use_cache=args.cache
    )

    # Modelo (base)
    num_classes = len(class_names)
    model, base = build_model(num_classes=num_classes, img_size=args.img_size)
    compile_model(model, lr=args.learning_rate)

    # Opcional: reanudar
    resume_path = None
    if args.resume_from:
        if args.resume_from.lower() == "auto":
            resume_path = _resolve_resume_path(Path(args.output_dir))
        else:
            rp = Path(args.resume_from)
            resume_path = rp if rp.exists() else None

        if resume_path:
            print(f"[Resume] Cargando pesos desde: {resume_path}")
            # Cargar modelo completo guardado (.keras)
            model = tf.keras.models.load_model(resume_path)
            # Reubicar referencia al backbone para FT
            try:
                base = _find_backbone(model)
            except Exception as e:
                print(f"[Resume] Advertencia: {e} — continuaré sin FT parcial preciso.")
            # Recompilar con LR de la fase actual:
            compile_model(model, lr=args.learning_rate)

    # Directorios/Callbacks
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / f"stanford-dogs_{ts}"
    ckpt_path = out_dir / "checkpoints" / "best.keras"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=max(3, args.epochs // 3) if args.epochs > 0 else 2,
            restore_best_weights=True
        )
    ]

    # Entrenamiento (fase congelada)
    if args.epochs and args.epochs > 0:
        print("[Train] Fase 1: backbone congelado")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1
        )

    # Fine-tuning (opcional)
    if args.fine_tune and args.fine_tune > 0:
        print("[Train] Fase 2: fine-tuning (parcial por defecto)")
        # Asegurar que tenemos referencia al backbone
        try:
            base = _find_backbone(model)
        except Exception:
            pass  # si no se encuentra, seguiremos con todo el modelo entrenable

        # Descongelado parcial (si ft_last > 0)
        if args.ft_last > 0 and base is not None:
            base.trainable = True
            for layer in base.layers[:-args.ft_last]:
                layer.trainable = False
        else:
            # full FT
            for l in model.layers:
                l.trainable = True

        # LR bajo para no romper pesos pre-entrenados
        compile_model(model, lr=min(args.ft_learning_rate, 5e-6))

        ft_callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=1, min_lr=1e-6
            )
        ]

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune,
            callbacks=callbacks + ft_callbacks,
            verbose=1
        )

    # Evaluación final
    print("[Eval] Evaluando en test…")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"[Eval] Test accuracy: {test_acc:.4f}")

    # --- Exportar pesos/formatos ---
    # 1) Cargar mejor checkpoint si existe (de esta sesión)
    if ckpt_path.exists():
        model = tf.keras.models.load_model(ckpt_path)

    # 2) Archivo .keras (ideal para FastAPI)
    keras_path = out_dir / "model.keras"
    model.save(keras_path, include_optimizer=False)
    print(f"[Export] Keras file: {keras_path}")

    # 3) Carpeta SavedModel (para TFLite/TF Serving)
    savedmodel_dir = out_dir / "saved_model"
    model.export(str(savedmodel_dir))
    print(f"[Export] SavedModel dir: {savedmodel_dir}")

    # 4) Etiquetas
    labels_path = out_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"[Export] labels.json: {labels_path}")

    print(f"[Export] best.keras:  {ckpt_path}")


def main():
    tf.random.set_seed(SEED)
    args = parse_args()
    print("[Args]", vars(args))
    train_and_export(args)


if __name__ == "__main__":
    main()
