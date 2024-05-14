from transformers import TFAutoModelForSequenceClassification, DistilBertConfig
from sklearn.model_selection import train_test_split
from tf_keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from transformers import BertTokenizerFast
from tensorflow.keras.layers import *
from tensorflow.keras import models
from transformers import pipeline
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# from tensorflow import device_lib
# print("CUDA:")
# print(device_lib.list_local_devices())

df = pd.read_csv('./rotulados.csv')
df_text = df['text']
df_label = df['label']
# train | val
train_texts, test_texts, train_labels, test_labels = train_test_split(df_text,df_label, test_size=.18,random_state=42)


# test
# test_texts = data_test['test_pt_original'].values.tolist()
# test_labels = data_test['test_labels'].values.tolist()

# # BERT model
#https://huggingface.co/transformers/v3.3.1/pretrained_models.html
# https://github.com/neuralmind-ai/portuguese-bert
#'neuralmind/bert-base-portuguese-cased' #'bert-base-multilingual-uncased'
bert = 'neuralmind/bert-base-portuguese-cased'

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# truncation=True corta os tokens que excederam o comprimento máximo permitido.
                    #Isso é útil ao lidar com textos muito longos para a entrada do modelo.
# padding = True completa as sequências para que todas tenham o mesmo comprimento.
# return_tensors define o formato dos outputs

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, return_tensors='np').data
val_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, return_tensors='np').data
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, return_tensors='np').data

print(train_encodings)

'''
for key, value in test_encodings.items():
    print(f"{key}: {value}")
'''

# # Fine-tuning
#TFAutoModelForSequenceClassification.from_pretrained
model = TFAutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2, id2label={0: 'non-ofensive', 1: 'ofensive'})
optimizer = tf.keras.optimizers.AdamW(learning_rate=2e-5)
model.compile(optimizer="adam", loss=model.hf_compute_loss)

print(model.summary())
# Dropout - Durante o treinamento, a camada de Dropout desativa aleatoriamente um certo percentual de neurônios
#           em cada iteração. Assim, reduz a dependência em neurônios específicos e torna a iteração mais rápida
#           porque menos neurônios são ativados.


# Definir o callback EarlyStopping para monitorar a perda (loss)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min', verbose=1)
maximo_epocas = 30
size = 2

# início tempo de treinamento
start_time = time.time()

# Treinamento com número maior de épocas e EarlyStopping
model.fit(train_encodings, np.array(train_labels),
          validation_data=(val_encodings, np.array(test_labels)),
          epochs=maximo_epocas, batch_size=size, callbacks=[early_stopping])

# fim tempo de treinamento
end_time = time.time()
execution_time = (end_time - start_time)/60
print(f"Tempo de execução: {execution_time} minutos")

#model.evaluate(test_encodings, np.array(test_labels))

#dir_model = "./bert_original_adamW"

#model.save_pretrained(dir_model)
#tokenizer.save_pretrained(dir_model)


# # Classifier
#import pandas as pd
#import numpy as np

dir_model = "./bert_original_adamW"
load_model=dir_model
load_tokenizer=dir_model

pipe = pipeline("text-classification", model=load_model,tokenizer=load_tokenizer)

pred_labels = pipe(test_texts)

bert_labels = [item['label'] for item in pred_labels]

# 1: fake; 0: opinion; -1 e/ou 2: news
bert_labels = [label.replace('Ofensivo', '1') for label in bert_labels]
bert_labels = [label.replace('De boa', '0') for label in bert_labels]

bert_labels = [int(item) for item in bert_labels]

print(classification_report(bert_labels, test_labels))