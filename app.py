

    
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

songs_list = pickle.load(open("songs.pkl","rb"))
data_1= pd.DataFrame(songs_list)

song_vectorizer = CountVectorizer()
song_vectorizer.fit(data_1['genre '])

# Assuming you have the necessary imports and data loaded already

# Function to get similarities
def get_similarities(song_name, data):
   
  # Getting vector for the input song.
  text_array1 = song_vectorizer.transform(data[data['name ']==song_name]['genre ']).toarray()
  num_array1 = data[data['name ']==song_name].select_dtypes(include=np.number).to_numpy()
   
  # We will store similarity for each row of the dataset.
  sim = []
  for idx, row in data.iterrows():
    name = row['name ']
     
    # Getting vector for current song.
    text_array2 = song_vectorizer.transform(data[data['name ']==name]['genre ']).toarray()
    num_array2 = data[data['name ']==name].select_dtypes(include=np.number).to_numpy()
 
    # Calculating similarities for text as well as numeric features
    text_sim = cosine_similarity(text_array1, text_array2)[0][0]
    num_sim = cosine_similarity(num_array1, num_array2)[0][0]
    sim.append(text_sim + num_sim)
     
  return sim
    # Your existing code for calculating similarities

# Function to recommend songs
def recommend_songs(song_name, data=data_1):
  # Base case
  if data_1[data_1['name '] == song_name].shape[0] == 0:
    print('This song is either not so popular or you\
    have entered invalid_name.\n Some songs you may like:\n')
    
    
     
    for song in data.sample(n=5)['name '].values:
      print(song)
    return
   
  data['similarity_factor'] = get_similarities(song_name, data)
 
  data.sort_values(by=['similarity_factor'],
                   ascending = [False],
                   inplace=True)
   
  # First song will be the input song itself as the similarity will be highest.
  return(data[[ 'name ']][2:7])

# Load your data (replace this with your actual data loading)
# data_1 = ...

# Streamlit app
def main():
    st.title("Song Recommender System")
    st.image('alan-walker-creative-logo-3x.jpg',  width=600)

    # User input for the song name
    song_name = st.selectbox(
    "Enter the name of a song:",
    data_1["name "].values)
    
    
    if st.button("Recommand", type="primary"):
        st.write('You selected:', song_name)

    # Check if the user has entered a song name
    if song_name:
        # Call the recommend_songs function
        recommendations = recommend_songs(song_name, data_1)

        # Display the recommendations in a Streamlit table
        st.table(recommendations)
    else:
        st.warning("Please enter a song name.")

if __name__ == "__main__":
    main()

