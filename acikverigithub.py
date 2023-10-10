import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import unidecode
from datetime import datetime

# Function to preprocess text and remove diacritics
def preprocess_text(text):
    return unidecode.unidecode(text.lower())

# Function to determine the appropriate greeting based on the time of day
def get_greeting():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    if "05:00:00" <= current_time < "12:00:00":
        return "Günaydın!"
    elif "12:00:00" <= current_time < "18:00:00":
        return "İyi günler!"
    else:
        return "İyi akşamlar!"

# Function to find similar data sets
def find_similar_data(user_question, data, df):
    found_data = []
    for item in data:
        if user_question.lower() in item['name'].lower() or user_question.lower() in item['explanation'].lower():
            found_data.append(item)
    return found_data

# Function to print similar data sets
def print_similar_data(similar_items):
    print("Aşağıdaki veri setlerinde eşleşme bulundu🤩:\n")
    for i, item in enumerate(similar_items):
        name, explanation, address = item
        print(f"{i + 1}. Veri seti: {name}, Açıklama: {explanation}, Link: {address}\n")

# Main function
def main():
    data = []

    df = pd.DataFrame(data)

    veri_istegi_url = 'weburl'

    timeout_seconds = 150  # 2,5 dakika süre
    max_wrong_attempts = 3  # Kullanıcının üç yanlış giriş hakkı

    wrong_attempts = 0  # Yanlış giriş sayacı

    # Karşılama mesajını belirle
    greeting = get_greeting()
    print(greeting)

    while True:
        start_time = time.time()

        user_question = input(
            "Merhaba, veri setlerini kolayca bulmanı sağlamak için buradayım !😊\n"
            "Lütfen ilgilendiğiniz konuyu tek kelime ile ifade edin:\n"
            "Örneğin: 'İzmirim Kart'\n\n"
            "Eğer veri talebinde bulunmak isterseniz 'Veri isteği' yazarak ilgili link üzerinden başvurabilirsiniz.\n"
            "('q' tuşuna basarak çıkabilirsiniz): "
        )

        if user_question.lower() == 'q':
            print("Size yardımcı olmaktan mutluluk duydum, tekrar görüşmek üzere.👋")
            break

        found_data = find_similar_data(user_question, data, df)

        if not found_data:
            tfidf_vectorizer = TfidfVectorizer(lowercase=True, preprocessor=preprocess_text)
            tfidf_matrix = tfidf_vectorizer.fit_transform(df["name"])

            user_question_tfidf = tfidf_vectorizer.transform([preprocess_text(user_question)])

            # Benzerlik hesapla
            similarities = cosine_similarity(user_question_tfidf, tfidf_matrix)
            similar_indices = [i for i, similarity in enumerate(similarities[0]) if similarity > 0.2]
            similar_items = [(df['name'][i], df['explanation'][i], df['address'][i]) for i in similar_indices]

            if not similar_items:
                wrong_attempts += 1
                if wrong_attempts < max_wrong_attempts:
                    print("Aranan veri seti bulunamadı.🙁 Lütfen daha spesifik bir kelime veya terim kullanmayı deneyin.\n\n")
                else:
                    print("🙁 Üzgünüm, ard arda yanlış girişler yaptınız sizlere yardım etmek için örnek veri seti konu başlıklarından bazılarını hazırladım:\n\n")
                    for i, item in enumerate(data[:5]):
                        print(f"{i + 1}. Veri seti: {item['name']}\n")
                    wrong_attempts = 0  # Yanlış giriş sayacını sıfırla ve döngüye devam et
            else:
                print_similar_data(similar_items)
        else:
            print("Aşağıdaki veri setlerinde eşleşme bulundu🤩:\n")
            for item in found_data:
                print(f"Konu Başlığı: {item['name']}\n")
                print(f"Açıklama: {item['explanation']}\n")
                print(f"Link: {item['address']}\n")

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            print("Otomatik çıkış: Süre aşıldı.😴")
            break

        if user_question.lower() == 'veri isteği':
            print(f"Veri isteği yapmak için bu linke başvurabilirsiniz: {veri_istegi_url}")

if __name__ == "__main__":
    main()
