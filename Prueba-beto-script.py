import pandas as pd

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

#### Controlar el error de verificacion de la API ######
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

session = requests.Session()
session.verify = False
#### Controlar el error de verificacion de la API ######

def get_embedding(text):
    # Tokenize and convert to input IDs
    inputs = tokenizer.encode_plus(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    # Forward pass, get hidden states
    outputs = model(**inputs)
    # Use the [CLS] embedding as the sentence embedding
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings


folder = 'data'

df_muestra = pd.read_excel(f"{folder}/Muestra.xlsx")
df_ocupaciones = pd.read_excel(f"{folder}/BaseCUOC.xlsx", sheet_name="Ocupacion")
df_descripciones = pd.read_excel(f"{folder}/BaseCUOC.xlsx", sheet_name="Descripcion")

tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Codificar en embeddings todas las ocupaciones. Se concatena el título y la descripción y se halla el embedding
ocupaciones = list()
for i in range(len(df_ocupaciones)):
  # Las tablas de Nombre ocupacion y descripción están sincronizadas en el excel
  titulo = df_ocupaciones.loc[i, "Nombre Ocupación"]
  descripcion = df_descripciones.loc[i, "Descripción Ocupación"]
  concatenado = (titulo+". "+descripcion).lower()
  ocupaciones.append({
      "id" : df_ocupaciones.loc[i, "Ocupación"],
      "titulo" : titulo,
      "descripcion" : descripcion,
      "concatenado" : concatenado, 
      "procesado" : get_embedding(concatenado)
  })


# Codificar en embeddings de los cargos de muestra. Se concatena el título y la descripción, y se halla el embedding
cargos = list()
for i in range(len(df_muestra)):
  titulo = df_muestra.loc[i,"Cargo"]
  descripcion = df_muestra.loc[i,"Descripción SP"]
  concatenado = (titulo+". " +descripcion).lower() 
  cargos.append({
          "titulo" : titulo,
          "descripcion" : descripcion,
          "concatenado" : concatenado,
          "procesado" : get_embedding(concatenado),
          "homologacion_manual" : { "id": df_muestra.loc[i, "Codigo Cuoc"], "titulo": df_muestra.loc[i, "Ocupación"]}
      })
  
# Comparar la similitud de cada cargo con todas las ocupaciones
for cargo in cargos:
  cargo["similarities"] = list()
  for ocupacion in ocupaciones:
    cargo_vec = cargo["procesado"].detach().numpy()
    ocupacion_vec = ocupacion["procesado"].detach().numpy()
    sim = cosine_similarity(cargo_vec, ocupacion_vec)[0][0]
    cargo["similarities"].append((sim, ocupacion["id"], ocupacion["titulo"]))
  
  cargo["similarities"].sort(reverse=True)


# Guardar los resultados
resultados = list()
for cargo in cargos:
  more_similar_oc_id = cargo["similarities"][0][1]
  similarity = cargo["similarities"][0][0]

  if more_similar_oc_id == cargo["homologacion_manual"]["id"]:
    correcto = True
  else:
    correcto = False
  
  resultados.append((cargo["titulo"], cargo["descripcion"], more_similar_oc_id, similarity, correcto))

column_names = ["titulo_cargo", "descripcion_cargo", "ocupacion_mas_similar", "similitud", "es_correcto"]
resultados = pd.DataFrame(resultados, columns=column_names)

resultados.to_excel("Resultados prueba beto.xlsx", index=False)