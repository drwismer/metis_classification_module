# ----------------- Import Libraries and Models ----------------- #

import streamlit as st

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util

import pandas as pd
import numpy as np
from collections import Counter

import pickle
from sklearn.ensemble import RandomForestClassifier

import plotly.graph_objects as go

# Load the random forest classifier
with open('pickle_rf_model.pickle', 'rb') as read_file:
    rf_classifier = pickle.load(read_file)

    
# ----------------- Spotify Connection ----------------- #

# Credentials
client_id = '_______________________________'
client_secret = '_______________________________'
username = '_______________________________'

# Scope Options:  https://developer.spotify.com/web-api/using-scopes/
scope = 'user-library-read playlist-modify-public playlist-read-private'

# This can be any valid url, but must be matched on your developer dashboard
redirect_uri = 'https://github.com/drwismer'

def sign_in():
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret) 
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)

    if token:
        return spotipy.Spotify(auth=token)
    else:
        print("Can't get token for", username)
        
sp = sign_in()


# ----------------- Spotify Retrieval Functions ----------------- #

def get_recommendations(sp, genres=[], artists=[], tracks=[]):
    """
    Generate a set of track URI's (Uniform Resource Indicator) for a given genre.
    Use the recommendations function to generate new lists of 100 songs (the upper
    limit) until the desired number of songs is found or you reach the maximum
    iterations specified. Return as list.
    """
    
    sp = sign_in() # Establish Spotify connection
    
    recs = sp.recommendations(seed_genres=genres, seed_artists=artists, seed_tracks=tracks, limit=100)['tracks']
    
    artist_names = []
    track_names = []

    for rec in recs:

        artists = []
        for a in rec['artists']:
            artists.append(a['name'])
        artists = ', '.join(artists)

        artist_names.append(artists)
        track_names.append(rec['name'])

    recs_df = pd.DataFrame(list(zip(track_names, artist_names)), columns =['Track', 'Artists'])
    
    track_list = [track['uri'] for track in recs]
    
    return track_list, recs_df

def tracks_from_playlists(sp, playlist_uri):
    """
    Return all track URI's for tracks in a given list of playlists by playlist URI.
    """
    
    sp = sign_in()
    
    results = sp.user_playlist_tracks(playlist_id=playlist_uri)
    playlist_tracks = results['items']
    
    # Loop to ensure all tracks are captured
    track_uris = []
    artist_names = []
    track_names = []
    
    while results['next']:
        results = sp.next(results)
        playlist_tracks.extend(results['items'])

    # Save playlist track URI's to list
    for item in playlist_tracks:
        track = item['track']
        try:
            track_uris.append(track['uri'])

            artists = []
            for a in track['artists']:
                artists.append(a['name'])
            artists = ', '.join(artists)

            artist_names.append(artists)
            track_names.append(track['name'])
        except TypeError:
            # Skip tracks that have no data
            pass

    playlist_df = pd.DataFrame(list(zip(track_names, artist_names)), columns =['Track', 'Artists'])
    
    return list(set(track_uris)), playlist_df

def get_audio_features(sp, track_list):
    """
    Accept track list and return dataframe with audio features.
    """
    
    sp = sign_in() # Establish Spotify connection
    
    track_list = list(track_list)
    
    audio_df = pd.DataFrame(sp.audio_features(track_list[0]))
    start = 1
    while start < len(track_list):
        end = min(start + 100, len(track_list))
        
        # For subsets with no audio data, loop through one by one and pass on tracks missing data
        try:
            audio_df = pd.concat([audio_df, pd.DataFrame(sp.audio_features(track_list[start:end]))])
        except AttributeError:
            for i in range(start, end):
                try:
                    audio_df = pd.concat([audio_df, pd.DataFrame(sp.audio_features(track_list[i]))])
                except AttributeError:
                    pass
            
        start += 100
    
    return audio_df.reset_index(drop=True)


# ----------------- EDM Genre Prediction Functions ----------------- #

api_cols = ['acousticness',
            'danceability',
            'duration_ms',
            'energy',
            'instrumentalness',
            'key',
            'liveness',
            'loudness',
            'mode',
            'speechiness',
            'time_signature',
            'tempo',
            'valence']

# Key and Time Signature not in use in this version
model_cols = ['acousticness',
              'danceability',
              'duration_ms',
              'energy',
              'instrumentalness',
              'liveness',
              'loudness',
              'speechiness',
              'tempo',
              'valence',
              #'key_0',
              #'key_1',
              #'key_2',
              #'key_3',
              #'key_4',
              #'key_5',
              #'key_6',
              #'key_7',
              #'key_8',
              #'key_9',
              #'key_10',
              #'key_11',
              #'time_sig_1',
              #'time_sig_3',
              #'time_sig_4',
              #'time_sig_5',
              'mode']

def impute_median(df):
    """
    Impute median for columns with NaN.
    """
    for col in df.columns[df.isna().any()].tolist():
        df[col] = df[col].fillna(df[col].median())
    
    return df

def dummies(df):
    """
    Add dummy variable columns for key and time signature.
    """
    
    for i in sorted(df['key'].unique()):
        df['key_' + str(int(i))] = np.where(df['key'] == i, 1, 0)
    
    for i in sorted(df['time_signature'].unique()):
        df['time_sig_' + str(int(i))] = np.where(df['time_signature'] == i, 1, 0)
        
    df.drop(columns=['key', 'time_signature'], inplace=True)
    
    return df

def build_model_data(track_list):
    """
    Accept audio features dataframe and configure for classification.
    """
    audio_df = get_audio_features(sp, track_list)
    
    # Drop and create columns required for the classification model
    audio_df = audio_df[api_cols]
    audio_df = impute_median(audio_df)
    #audio_df = dummies(audio_df)  # no dummification required in this version
    missing = [col for col in model_cols if not col in list(audio_df.columns)]

    if missing:
        for col in missing:
            audio_df[col] = 0

    audio_df = audio_df[model_cols]
    
    return audio_df

def make_predictions(audio_df, rf_classifier):
    """
    Accept audio features dataframe and return genre percentages.
    """
    
    # Predict genres for the Spotify recommendations (100 songs)
    predictions = rf_classifier.predict(audio_df)
    pred_count = Counter(predictions)
    
    # Return summary dataframe
    pred_df = pd.DataFrame({'Genre' : ['Deep House',
                                       'Drum\'n\'Bass',
                                       'Dubstep',
                                       'Hardstyle',
                                       'Progressive House',
                                       'Techno',
                                       'Trance'],
                            'Counts' : [pred_count['deep_house'],
                                        pred_count['dnb'],
                                        pred_count['dubstep'],
                                        pred_count['hardstyle'],
                                        pred_count['prog_house'],
                                        pred_count['techno'],
                                        pred_count['trance']]
                           }).sort_values('Counts', ascending=False)
    
    pred_df['Percentage'] = pred_df['Counts']/pred_df['Counts'].sum()
    
    replace_dict = {'deep_house' : 'Deep House',
                    'dnb' : 'Drum\'n\'Bass',
                    'dubstep' : 'Dubstep',
                    'hardstyle' : 'Hardstyle',
                    'prog_house' : 'Progressive House',
                    'techno' : 'Techno',
                    'trance' : 'Trance'}
    
    # Make predictions column more readable
    predictions = [replace_dict[pred] for pred in predictions]
    
    return pred_df[['Genre', 'Percentage']].reset_index(drop=True), predictions

def predict_track(audio_df, rf_classifier):
    """
    Accept audio features dataframe and return genre percentages.
    """
    
    # Predict genres for the Spotify recommendations (100 songs)
    predictions = rf_classifier.predict_proba(audio_df).flatten()
    
    # Return summary dataframe
    pred_df = pd.DataFrame({'Genre' : ['Deep House',
                                       'Drum\'n\'Bass',
                                       'Dubstep',
                                       'Hardstyle',
                                       'Progressive House',
                                       'Techno',
                                       'Trance'],
                            'Percentage' : predictions
                           }).sort_values('Percentage', ascending=False)
    
    return pred_df.reset_index(drop=True)


# ----------------- Consolidated Functions ----------------- #

def seeds_to_classification(sp, rf_classifier, genres=[], artists=[], tracks=[]):
    track_list, recs_df = get_recommendations(sp, genres=genres, artists=artists, tracks=tracks)
    audio_df = build_model_data(track_list)
    summary_df, predictions = make_predictions(audio_df, rf_classifier)
    recs_df['Genre Prediction'] = predictions
    return summary_df, recs_df

def playlist_to_classification(sp, rf_classifier, playlist_uri):
    playlist_tracks, playlist_df = tracks_from_playlists(sp, playlist_uri)
    audio_df = build_model_data(playlist_tracks)
    summary_df, predictions = make_predictions(audio_df, rf_classifier)
    playlist_df['Genre Prediction'] = predictions
    return summary_df, playlist_df

def track_to_classification(sp, rf_classifier, track_uri):
    audio_df = build_model_data([track_uri])
    summary_df = predict_track(audio_df, rf_classifier)
    return summary_df


# ----------------- Streamlit Formatting ----------------- #

# Wide layout
st.set_page_config(layout='wide')

# Gradient background
st.markdown("""
<style>section[data-testid="stSidebar"] div[class="css-17eq0hr e1fqkh3o1"] {background-image: linear-gradient(#000000,#000000);color: white}
</style>""",unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .reportview-container {
        background-image: linear-gradient(to bottom right, #000000, #110022, #330E55, #804DB0, #E7CEFF);
        color: #110022;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dataframe styling

df_style = {'background-color': 'white', 'color' : 'black', 'text-align': 'left'}


# ----------------- Streamlit Functions ----------------- #

def show_plotly(df, summary=True):
    series_list = [df[col] for col in df]
    fig = go.Figure(data=[go.Table(header=dict(values=list(df.columns),
                                               fill_color='#1cc5d4',
                                               align='left',
                                               font=dict(color='#000000', size=16),
                                               line=dict(color='#000000'),
                                               height=40),
                                   cells=dict(values=series_list,
                                              format=[None, '.0%'] if summary else [],
                                              fill_color='#FFFFFF',
                                              align='left',
                                              font=dict(color='#000000', size=14),
                                              line=dict(color='#000000'),
                                              height=40)
                                  )])
    
    fig.update_layout(margin=dict(l=5, r=5, t=10), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    

# ----------------- Streamlit Page Construction ----------------- #

pages = st.sidebar.selectbox('Choose your Genrify tool:',
                             ('Recommendation Classifier', 'Playlist Classifier', 'Track Classifier'))
st.sidebar.write('---')
st.sidebar.write("""The text inputs in the Genrify form require that you paste in track, artist, and playlist **URI's**. In your Spotify 
                 app, simply right click on the item you wish to copy and go to the 'Share' option. Hold down the 'Option' key on Mac 
                 or the 'Alt' key on Windows, and you will see the 'Copy Spotify URI' option appear! For a quick video, scroll to the 
                 bottom of this [article](https://community.spotify.com/t5/FAQs/What-s-a-Spotify-URI/ta-p/919201).""")
st.sidebar.write('---')
st.sidebar.write('Genrify was built by me, David Wismer. Find me on [LinkedIn](https://www.linkedin.com/in/david-wismer-0a940656/).')
st.sidebar.write('---')
st.sidebar.write("""Visit my [Github](https://github.com/drwismer) to see how the app was built. You'll also find notebooks detailing data 
                 collection, EDA, and the modeling process.""")

available_genres = ("edm",
                    "deep-house",
                    "drum-and-bass",
                    "dubstep",
                    "hardstyle",
                    "progressive-house",
                    "techno",
                    "trance",
                    "------------------------------------------------------",
                    "acoustic",
                    "alt-rock",
                    "alternative",
                    "bluegrass",
                    "blues",
                    "breakbeat",
                    "classical",
                    "country",
                    "dance",
                    "dancehall",
                    "death-metal",
                    "disco",
                    "folk",
                    "funk",
                    "gospel",
                    "goth",
                    "grunge",
                    "hardcore",
                    "heavy-metal",
                    "hip-hop",
                    "house",
                    "indie",
                    "indie-pop",
                    "jazz",
                    "k-pop",
                    "latino",
                    "metal",
                    "opera",
                    "pop",
                    "power-pop",
                    "punk",
                    "punk-rock",
                    "r-n-b",
                    "reggae",
                    "reggaeton",
                    "rock",
                    "rock-n-roll",
                    "salsa",
                    "samba",
                    "ska",
                    "soul",
                    "tango",
                    "trip-hop")

if pages == 'Recommendation Classifier':
    
    st.image('app_images/genrify.png', width=400)

    left, middle, right = st.beta_columns([1, 0.5, 1])

    with left:
        with st.form(key='seed_inputs'):
            st.write('## Spotify Recommendations Genrifier')
            st.write('#### Provide up to 5 seeds (ex: 2 genres, 2 artists, 1 track)\n')
            genre_inputs = st.multiselect('Genres', available_genres)
            artist_inputs = st.text_area("Artist URI's - Separate with commas")
            track_inputs = st.text_area("Track URI's - Separate with commas")
            submit_recs = st.form_submit_button('Genrify!', )

    if submit_recs:
        artists = [a.strip() for a in artist_inputs.split(',')]
        tracks = [t.strip() for t in track_inputs.split(',')]

        if len(genre_inputs) + len(artists) + len(tracks) > 5:
            left.error('Spotify only allows up to 5 seed inputs. Reduce the genres, artists, and tracks to no more than 5!')
        else:
            try:
                seed_df, recs = seeds_to_classification(sp, rf_classifier, genres=genre_inputs, artists=artists, tracks=tracks)

                with middle:
                    st.write('#### Genre Breakdown')
                    show_plotly(seed_df, summary=True)
                with right:
                    st.write('#### Recommended Tracks - Scroll or expand to see more')
                    show_plotly(recs, summary=False)
                    #st.dataframe(recs.style.set_properties(**df_style))
            except:
                left.error('One of the URI\'s provided is not valid! Please try again.')


elif pages == 'Playlist Classifier':
    
    st.image('app_images/genrify.png', width=400)

    left, middle, right = st.beta_columns([1, 0.5, 1])

    with left:
        with st.form(key='playlist_input'):
            st.write('## Spotify Playlist Genrifier')
            st.write('#### Provide a single playlist URI\n')
            playlist_input = st.text_input('')
            submit_playlist = st.form_submit_button('Genrify!')
            
    if submit_playlist:
        try:
            playlist_df, playlist_detail = playlist_to_classification(sp, rf_classifier, playlist_input)
            
            with middle:
                st.write('#### Genre Breakdown')
                show_plotly(playlist_df, summary=True)
            with right:
                st.write('#### Playlist Tracks - Scroll or expand to see more')
                show_plotly(playlist_detail, summary=False)
        
        except:
            left.error('The playlist URI you provided does not appear to be valid! Please try again.')

elif pages == 'Track Classifier':
    
    st.image('app_images/genrify.png', width=400)

    left, middle, right = st.beta_columns([1, 0.5, 1])

    with left:
        with st.form(key='playlist_input'):
            st.write('## Spotify Track Genrifier')
            st.write('#### Provide a single track URI\n')
            track_input = st.text_input(' ')
            submit_track = st.form_submit_button('Genrify!')
            
    if submit_track:
        try:
            track_df = track_to_classification(sp, rf_classifier, track_input)

            with middle:
                st.write('#### Genre Breakdown')
                show_plotly(track_df, summary=True)
        
        except:
            left.error('The track URI you provided does not appear to be valid! Please try again.')
