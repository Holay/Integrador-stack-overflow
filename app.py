from flask import Flask, request, jsonify
import boto3
import pandas as pd
import io
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
import os

app = Flask(__name__)

s3_client = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
model = SentenceTransformer('all-MiniLM-L6-v2')
def convert(item):
    item = item.strip()  
    item = item[1:-1]    
    item = np.fromstring(item, sep=' ')  
    return item

def generate_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings

bucket_name = 'integrador-stack-overflow'
file_key = 'embedded_100k_reviews.csv'

obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
data = obj['Body'].read()
df = pd.read_csv(io.BytesIO(data))
# df = pd.read_csv('./output/embedded_100k_reviews.csv')
print('Datos cargados')
print(len(df))
df['ada_embedding'] = df['ada_embedding'].apply(convert)

def find_similar_questions(query_embedding, embeddings, top_k=10):
    similarities = cosine_similarity([query_embedding], embeddings)
    similar_indices = np.argsort(-similarities[0])[:top_k]
    return similar_indices

def filter_dataframe_by_tags(df, tags):

    original_row_count = len(df)
    filtered_df = df[df['QuestionTags'].apply(lambda row_tags: any(tag in row_tags for tag in tags))]
    filtered_row_count = len(filtered_df)
    
    print(f"Se filtraron {original_row_count - filtered_row_count} filas. Numero de filas antes del filtro: {original_row_count}, Numero de filas filtradas: {filtered_row_count}")
    
    return filtered_df

@app.route('/suggest-users', methods=['POST'])
def index():
    body = request.get_json()
    
    if not body:
        return jsonify({"error": "Invalid input"}), 400
    
    question = body.get('question')
    tags = body.get('tags')
    print(tags)
    new_embedding = generate_embeddings(question)
    filtered_data = filter_dataframe_by_tags(df, tags)
    # filtered_data
    similar_question_indices = find_similar_questions(new_embedding, np.array(filtered_data['ada_embedding'].tolist()))
    result_df = filtered_data.iloc[similar_question_indices][['QuestionBody', 'AnswerOwnerUserId']]
    
    result = result_df.to_dict(orient='records')

    for item in result:
        item['userURL'] = f"https://stackoverflow.com/users/{item['AnswerOwnerUserId']}"
    return jsonify(result)

if __name__ == '__main__':
    app.run()