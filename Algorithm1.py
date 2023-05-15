
import csv
import string
import math
import streamlit
from nltk.stem import PorterStemmer


# Read the data
movies_data = []
with open('tmdb_5000_movies.csv', 'r', encoding='utf-8') as movies_file:
    movies_reader = csv.reader(movies_file)
    for row in movies_reader:
        movies_data.append(row)


movies_data = movies_data[1:]  # remove header row


def tokenize(text):
    return text.lower().split()


def remove_punctuation(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    return ''.join(char for char in text if not char.isdigit())


def remove_stopwords(text):
    stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
                  'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
    words = text.lower().split()
    return ' '.join([word for word in words if word not in stop_words])

ps = PorterStemmer()
def stem(text):
    for count, word in enumerate(text):
        text[count] = ps.stem(text[count])
    return text

def preprocess(text):
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = tokenize(text)
    return stem(text)


def create_combined_features(movie):
    return preprocess(movie[3]) + preprocess(movie[4]) + preprocess(movie[1]) + preprocess(movie[13]) + tokenize(movie[2]) + preprocess(
        movie[6]) + tokenize(movie[7]) + tokenize(str(movie[11])) # + tokenize(str(movie[15]))


def calculate_tf(corpus, document):
    document_words = set(document)
    tf = {}
    for word in document_words:
        if document.count(word) > 0:
            tf[word] = 1+math.log10(document.count(word))
        else:
            tf[word] = 0
    return tf


def calculate_idf(corpus):
    idf = {}
    num_documents = len(corpus)
    for document in corpus:
        for word in document:
            if word not in idf:
                idf[word] = 0
            idf[word] += 1

    for word, count in idf.items():
        idf[word] = math.log(num_documents / count)
    return idf


def calculate_tfidf(corpus):
    idf = calculate_idf(corpus)
    tfidf = []
    for document in corpus:
        tf = calculate_tf(corpus, document)
        tfidf_document = {word: tf[word] * idf[word] for word in document}
        tfidf.append(tfidf_document)
    return tfidf


def cosine_similarity(tfidf1, tfidf2):
    intersection = set(tfidf1.keys()) & set(tfidf2.keys())
    numerator = sum([tfidf1[word] * tfidf2[word] for word in intersection])

    sum1 = sum([tfidf1[word] ** 2 for word in tfidf1.keys()])
    sum2 = sum([tfidf2[word] ** 2 for word in tfidf2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return numerator / denominator


def recommend_movies(input_movie_id):
    input_movie = None
    for movie in movies_data:
        if int(movie[0]) == int(input_movie_id):
            input_movie = movie
            break
    if input_movie is None:
        return None, []

    movie_features = [create_combined_features(movie) for movie in movies_data]
    tfidf_scores = calculate_tfidf(movie_features)

    input_movie_index = -1
    for index, movie in enumerate(movies_data):
        if int(movie[0]) == int(input_movie_id):
            input_movie_index = index
            break

    input_movie_tfidf = tfidf_scores[input_movie_index]
    similarities = []

    for index, movie_tfidf in enumerate(tfidf_scores):
        if index == input_movie_index:
            continue

        similarity = cosine_similarity(input_movie_tfidf, movie_tfidf)
        similarities.append((movies_data[index], similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return input_movie, similarities[:5]  # return input_movie and top 5 recommendations




# Given_Movie = streamlit.selectbox("Select Movie: ",movie_name_list_copy)


movie_name_ids = {}
movie_names = []
for movies in movies_data:
  movie_name_ids[movies[3]] = movies[0]
  movie_names.append(movies[3])
movie_names.sort()

input_movie_name = streamlit.selectbox("Select Movie: ",movie_names)
print(input_movie_name)
input_movie_id = movie_name_ids[input_movie_name]
print(input_movie_id)

#input_movie_id = 19995  # choose a movie to get recommendations for
input_movie, recommendations = recommend_movies(input_movie_id)

if input_movie:
    print(f"Recommendations for {input_movie[3]}:")
    streamlit.write("Recommendations for",input_movie[3],":")
    for movie, similarity in recommendations:
        print(f"{movie[3]} (similarity: {similarity:.4f})")
        streamlit.write(f"{movie[3]} (similarity: {similarity:.4f})")
else:
    print("Movie not found.")
    streamlit.write("Movie not found")