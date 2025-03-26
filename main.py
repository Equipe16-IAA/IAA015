# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ollama
import numpy as np
from numpy import dot
from numpy.linalg import norm


def helloworld():
    resp = ollama.chat(model='deepseek-r1:14b',messages=[
            {
                'role':'user',
                'content':'Quanto é 2+2?',
            },])
    print(resp['message']['content'])

#helloworld()

documentos = [
    { "id":1,"text":"a buzina azul do carro amarelo.",},
    { "id":2,"text":"Dennis Ritchie e Brian Kernighan criaram a linguagem de programação C.",},
    {"id":3,"text":"o documento azul autoriza o acesso a ala segura."},
    {"id":4, "text": "a buzina azul do carro vermelho.", },
]

doc_embeds = {
    doc['id']: ollama.embed(model='deepseek-r1:14b',input=doc['text']) for doc in documentos
}

def mostraEmbed():
    print( dir(doc_embeds[1]) )
    """[...
     'construct', 'copy', 'created_at', 'dict', 'done', 'done_reason', 'embeddings', 'eval_count', 'eval_duration',
     'from_orm', 'get', 'json', 'load_duration', 'model', 'model_computed_fields', 'model_config', 'model_construct', 
     'model_copy', 'model_dump', 'model_dump_json', 'model_extra', 'model_fields', 'model_fields_set', 
     'model_json_schema', 'model_parametrized_name', 'model_post_init', 'model_rebuild', 'model_validate', 
     'model_validate_json', 'model_validate_strings', 'parse_file', 'parse_obj', 'parse_raw', 'prompt_eval_count', 
     'prompt_eval_duration', 'schema', 'schema_json', 'total_duration', 'update_forward_refs', 'validate']"""
    print( doc_embeds[1].json )

mostraEmbed()

def euclidiana(a,b):
    return np.linalg.norm(a - b)

def cosvec(a,b):
   return dot(a, b)/(norm(a)*norm(b))

# vetores próximos
v1 = [ 1, 2 ]
v2 = [ 1, 3 ]
print("proximos: ", cosvec(v1,v2) )

# vetores iguais, mas em sentidos opostos
v1 = [ 1, 2 ]
v2 = [ -1, -2 ]
print("sentidos opostos: ",cosvec(v1,v2) )

# perpendicular (90 graus) - o "mais distante"
v1 = [ 0, 1 ]
v2 = [ 1, 0 ]
print("perpendiculares: ",cosvec(v1,v2) )

#
# dois doc embed
v1 = np.array(doc_embeds[1]["embeddings"][0])
v2 = np.array(doc_embeds[2]["embeddings"][0])
v3 = np.array(doc_embeds[3]["embeddings"][0])
v4 = np.array(doc_embeds[4]["embeddings"][0])
print("cosseno")
print("doc 1 e 2: ",cosvec(v1,v2) )
print("doc 1 e 3: ",cosvec(v1,v3) )
print("doc 2 e 3: ",cosvec(v2,v3) )
print("doc 1 e 4: ",cosvec(v1,v4) )

print("euclidiana")
print("doc 1 e 2: ",euclidiana(v1,v2) )
print("doc 1 e 3: ",euclidiana(v1,v3) )
print("doc 2 e 3: ",euclidiana(v2,v3) )
print("doc 1 e 4: ",euclidiana(v1,v4) )

def query_rag(query):
  query_embed = ollama.embed(model='deepseek-r1:14b',input=query)
  melhor_doc = None
  melhor_score = float("-inf")
  v_query = np.array(query_embed["embeddings"][0])
  for idx in doc_embeds:
      doc = doc_embeds[idx]
      v_doc = np.array(doc["embeddings"][0])
      score = cosvec(v_query,v_doc)
      if score>melhor_score:
        melhor_score=score
        melhor_doc = documentos[idx]
  return melhor_doc

prompt = "buzina vermelho"
busca_local = query_rag(prompt)
print("Melhor doc: ")
print(busca_local)

#
# pede para o modelo responder com base nos documentos
#
SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
                        based on the snippets of text provided in context. Answer
                        only using the context provided, being as concise as possible.
                        If you are unsure, just say you don't know.
                        Write in portuguese.
                        Context:
                        """

res = ollama.chat(model='deepseek-r1:14b',
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT +"\n".join( busca_local )
            },
            {
                "role": "user", "content": prompt
            }
        ]
    )
print(res["message"]["content"])

