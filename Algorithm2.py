import csv
import math
import string
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
    return set(text.lower().split())


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
    return set(text)

def preprocess(text):
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = tokenize(text)
    return set(stem(list(text)))


def create_combined_features(movie):
    return (preprocess(movie[1]) | preprocess(movie[3]) | preprocess(movie[4]) | tokenize(movie[2]) |
            preprocess(movie[6]) | tokenize(str(movie[7])) | preprocess(movie[13]) | preprocess(movie[11]) | preprocess(movie[12]))

count =0
def cosine_similarity(movie1, movie2):
    intersection = len(movie1.intersection(movie2))
    global count
    if count ==0:
        print(movie1, movie2)
        count+=1
    return intersection / (math.sqrt(len(movie1)) * math.sqrt(len(movie2)))


def recommend_movies(input_movie_id):
    input_movie = None
    for movie in movies_data:
        if int(movie[0]) == int(input_movie_id):
            input_movie = movie
            break
    if input_movie is None:
        return None, []

    input_features = create_combined_features(input_movie)
    similarities = []

    for movie in movies_data:
        if int(movie[0]) == int(input_movie_id):
            continue

        movie_features = create_combined_features(movie)
       # print(movie_features)
        similarity = cosine_similarity(input_features, movie_features)
        similarities.append((movie, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return input_movie, similarities[:5]  # return input_movie and top 5 recommendations


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