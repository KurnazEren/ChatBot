from concurrent.futures import ThreadPoolExecutor
import re
from flask import request_finished
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import unidecode
from datetime import datetime
import json
from quart import Quart, jsonify, request
import requests
from a2wsgi import ASGIMiddleware


app = Quart(__name__)
wsgi_app = ASGIMiddleware(app)

app.json.sort_keys = False

packages_url = 'https://acikveri.bizizmir.com/api/3/action/package_list'
detail_url = 'https://acikveri.bizizmir.com/api/3/action/package_show?id='
last_update_time = 0
update_time_interval = 43200 # 12 hours

main_data = []
df = pd.DataFrame()
tfidf_matrix = []

# Function to preprocess text and remove diacritics
def preprocess_text(text):
    return unidecode.unidecode(text.lower())


tfidf_vectorizer = TfidfVectorizer(lowercase=True, preprocessor=preprocess_text)

# fill data from api

def fill_data_from_api():
    global last_update_time, main_data, df, tfidf_matrix
    response = requests.get(packages_url, verify=False)
    page_lists = response.json()['result']
    
    def Get_detail_of_page(page_name):
        page_url = 'https://acikveri.bizizmir.com/api/3/action/package_show?id=' + page_name
        response = requests.get(page_url,verify=False)
        page_detail = response.json()['result']
        
        return page_detail
    
    results = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        resultsThreaded = executor.map(Get_detail_of_page, page_lists)
        
        for result in resultsThreaded:
            results.append({'name':result.get('title'),'explanation':result.get('notes'),'address':result.get('url')})

    df = pd.DataFrame(results)
    main_data = results
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["name"])
    last_update_time = time.time()

def get_greeting():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if "05:00:00" <= current_time < "12:00:00":
        greeting_message = "Günaydın!"
    elif "12:00:00" <= current_time < "18:00:00":
        greeting_message = "İyi günler!"
    else:
        greeting_message = "İyi akşamlar!"
    return greeting_message

@app.route('/greet', methods=['GET'])
def index():
    greeting_message = get_greeting()
    message = (
        "Merhaba, İzmir Büyükşehir Belediyesi Açık Veri Portalı içerisinde veri setlerini kolayca bulmanı sağlamak için buradayım !😊\n"
        "Lütfen ilgilendiğiniz konuyu tek kelime ile ifade edin:\n"
        "Örneğin: 'İzmirim Kart'\n\n"
        "Eğer veri talebinde bulunmak isterseniz ana menü üzerinde yer alan 'Veri isteği' alanından bizlere istek gönderebilirsiniz.\n"
    )
    return jsonify({"greeting": greeting_message, "message": message})





# Function to find similar data sets
def find_similar_data(user_question, data):
    found_data = []
    for item in data:
        if user_question.lower() in item['name'].lower() or user_question.lower() in item['explanation'].lower():
            found_data.append(item)
    return found_data


# Function to provide example data
def provide_example_data():
    response = []
    for i, item in enumerate(main_data[:5]):
        response.append({
            "Konu_Basligi": item['name']
        })

    return response

# Function to handle data request
def handle_data_request():
    veri_istegi_url = "https://acikveri.bizizmir.com/tr/datarequest/new"
    return {"message": f"Veri isteği yapmak için bu linke başvurabilirsiniz: {veri_istegi_url}"}

# Öncelikle bu kayıtları tutmak için bir liste tanımlayın
request_history = []

timeout_seconds = 150  # 2,5 dakika süre
max_wrong_attempts = 3  # Kullanıcının üç yanlış giriş hakkı

def RequestCountExceeded(user_agent, remote_addr):
    # remove all requests older than timeout 
    request_history[:] = [request for request in request_history if request['last_request'] and request['last_request'] + timeout_seconds > time.time()]
    
    # check if user has made too many wrong attempts
    user = [request for request in request_history if request['user_agent'] == user_agent and request['remote_addr'] == remote_addr]
    if not user:
        request_history.append({"user_agent": user_agent, "remote_addr": remote_addr, "count": 1, "last_request": time.time()})
        return False
    else:
        user[0]['count'] += 1
        if user[0]['count'] > max_wrong_attempts:
            request_history[:] = [request for request in request_history if request['user_agent'] != user_agent and request['remote_addr'] != remote_addr]
            return True
        else:      
            return False  


# Main route for API

@app.route('/api', methods=['POST'])
async def api():
    
    # check last update time and update data if needed
    if time.time() - last_update_time > update_time_interval:
        fill_data_from_api()
        
    
    
    req_json = await request.get_json()
    user_question = req_json.get('user_question')
    if user_question is None:
        return jsonify({"message": "Lütfen bir soru sorunuz."})
    
    # remove word starts or contains 'veri' in user question
    user_question = re.sub(r'\bveri\w*\b', '', user_question, flags=re.IGNORECASE)
    if(user_question == ' ') or (user_question == ''):
        return jsonify({"message": "Lütfen bir soru sorunuz."})
    
    
    user_agent = request.headers.get('User-Agent')
    client_ip = request.remote_addr

    if user_question.lower() == 'q':
        return jsonify({"message": "Size yardımcı olmaktan mutluluk duydum, tekrar görüşmek üzere.👋"})

    found_data = find_similar_data(user_question, main_data)

    if not found_data:
        user_question_tfidf = tfidf_vectorizer.transform([preprocess_text(user_question)])
        # Benzerlik hesapla
        similarities = cosine_similarity(user_question_tfidf, tfidf_matrix)
        similar_indices = [i for i, similarity in enumerate(similarities[0]) if similarity > 0.2]
        similar_items = [(df['name'][i], df['explanation'][i], df['address'][i]) for i in similar_indices]

        if not similar_items:
            if RequestCountExceeded(user_agent, client_ip):
                return jsonify({"message": "Sanırım bir sorun yaşıyorsunuz, Başlıklar ile eşleşen bir kayıt bulamıyorum.:", "Answer": provide_example_data()})
            else:
                return jsonify({"message": "Aranan veri seti bulunamadı.🙁 Lütfen daha spesifik bir kelime veya terim kullanmayı deneyin."})
                
                
        response = []
        for item in similar_items:
            name, explanation, address = item
            response.append({
                "Veri_Seti": name,
                "Link": address
            })
    else:
        response = []
        for item in found_data:
            response.append({
                "Konu_Basligi": item['name'],
                "Link": item['address']
            })
            
    return jsonify({"Answer": response})

if __name__ == "__main__":
    fill_data_from_api()
    
    app.run(debug=False, port=5650,host='0.0.0.0')
