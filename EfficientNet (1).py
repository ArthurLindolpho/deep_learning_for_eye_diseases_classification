import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import os

# --- Configurações Iniciais ---
DATASETS = {
    'Dataset_normal': 'dataset',
    'Dataset_Aumentado': 'dataset processed'
}

NOME_ATUAL = 'Dataset_Aumentado'
DATA_DIR = DATASETS[NOME_ATUAL]

# Cria a pasta de saída ANTES do loop para salvar os logs individuais
CAMINHO_SAIDA = 'resultado'
os.makedirs(CAMINHO_SAIDA, exist_ok=True)

print(f"Rodando experimentos para: {NOME_ATUAL}")
print(f"Lendo de: {DATA_DIR}")
print(f"Salvando resultados em: {CAMINHO_SAIDA}")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 4
SEEDS = [124, 546, 983, 23, 15]

def get_dataset(subset, seed=42):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal'],
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True if subset != 'test' else False,
        seed=seed,
        validation_split=0.2,
        subset=subset
    )

metricas_finais = []

for i, seed in enumerate(SEEDS):
    print(f"\n=== Execução {i+1}/5 (SEED: {seed}) - {NOME_ATUAL} ===")
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Carregamento dos dados
    train_ds = get_dataset('training', seed=seed)
    rest_ds = get_dataset('validation', seed=seed)
    
    val_batches = tf.data.experimental.cardinality(rest_ds)
    test_ds = rest_ds.take(val_batches // 2)
    val_ds = rest_ds.skip(val_batches // 2)
    
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Construção do Modelo
    inputs = layers.Input(shape=(224, 224, 3))
    
    base_model = EfficientNetB0(
        include_top=False, 
        weights=None, 
        input_tensor=inputs,
        pooling='avg'
    )
    
    output = layers.Dense(NUM_CLASSES, activation='softmax')(base_model.output)
    model = models.Model(inputs=inputs, outputs=output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # --- TREINAMENTO E CAPTURA DE LOGS ---
    # O objeto 'history' contém os dados de loss e accuracy de cada época
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

    # 1. Cria um DataFrame com o histórico (accuracy, loss, val_accuracy, val_loss)
    epoch_log_df = pd.DataFrame(history.history)
    
    # 2. Adiciona a coluna de época (começando de 1)
    epoch_log_df.insert(0, 'epoch', range(1, len(epoch_log_df) + 1))
    
    # 3. Salva o CSV individual desta Seed
    log_filename = os.path.join(CAMINHO_SAIDA, f'metrics_Seed_{seed}.csv')
    epoch_log_df.to_csv(log_filename, index=False)
    print(f"Log de treinamento por época salvo em: {log_filename}")

    # --- AVALIAÇÃO FINAL ---
    print("Calculando métricas finais...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    metricas_finais.append({
        'seed': seed,
        'accuracy': report['accuracy'],
        'precision': report['macro avg']['precision'],
        'recall': report['macro avg']['recall'],
        'f1-score': report['macro avg']['f1-score'],
        'confusion_matrix': cm
    })

# --- SALVAMENTO DOS RESUMOS GERAIS ---
df = pd.DataFrame(metricas_finais)
df_csv = df[['seed', 'accuracy', 'precision', 'recall', 'f1-score']].copy()
df_csv.loc['Média'] = df_csv.drop(columns=['seed']).mean()      
df_csv.loc['Desvio Padrão'] = df_csv.drop(columns=['seed']).std()

arquivo_csv_final = os.path.join(CAMINHO_SAIDA, f'metricas_finais_{NOME_ATUAL}.csv')
df_csv.to_csv(arquivo_csv_final)

arquivo_txt_matrizes = os.path.join(CAMINHO_SAIDA, f'matrizes_{NOME_ATUAL}.txt')
with open(arquivo_txt_matrizes, 'w') as f:
    f.write(f"Matrizes de Confusão: {NOME_ATUAL}\n\n")
    for item in metricas_finais:
        f.write(f"SEED {item['seed']}:\n{item['confusion_matrix']}\n\n")

print(f"\nTodos os resultados salvos na pasta: {CAMINHO_SAIDA}")