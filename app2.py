import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from flask import Flask, render_template, send_file
from io import BytesIO
import base64
from fuzzywuzzy import fuzz
from geopy import Nominatim
import pycountry
import requests
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import numpy as np
import os
from werkzeug.utils import secure_filename
import mysql.connector
from PIL import Image, ImageOps

app = Flask(__name__)

def get_mysql_connection():
    return mysql.connector.connect(
        user='root',
        password='',
        host='localhost',
        port=3306,
        database='ml'
    )

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model with the updated input shape
model = load_model("keras_Model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as labels_file:
    class_names = labels_file.readlines()

# Define the upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Google Custom Search API key and engine ID
API_KEY = 'AIzaSyBPBzwEUvgHZoK6ql6XDqec_QFVMtYJPo0'
SEARCH_ENGINE_ID = '02a5a8a3f399c4e05'


def get_extra_images(query, num_images=3):
  url = f'https://www.googleapis.com/customsearch/v1?q={query}&num={num_images}&key={API_KEY}&cx={SEARCH_ENGINE_ID}'
  response = requests.get(url)
  data = response.json()

  if 'items' in data:
    # Extract image URLs from the API response
    image_urls = [item['link'] for item in data['items']]
    return image_urls

  # Return default image URLs or an empty list
  return ["default_image1.jpg", "default_image2.jpg", "default_image3.jpg"]

def predict_class(image_path):
  # Load the selected image file
  image = Image.open(image_path).convert("RGB")

  # resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  # turn the image into a numpy array
  image_array = np.asarray(image)

  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  # Load the image into the array with the updated shape
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  data[0] = normalized_image_array

  # Predict the model
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index][2:]
  confidence_score = prediction[0][index]

  return class_name, confidence_score


def store_image_info(filename, class_name, confidence_score):
  connection = None  # Initialize connection outside the try block

  try:
    # Establish a MySQL connection
    connection = get_mysql_connection()

    # Create a cursor object
    with connection.cursor() as cursor:
      # Read image file data as bytes
      with open(filename, 'rb') as file:
        image_data = file.read()

      # Define the SQL query to insert image information
      query = "INSERT INTO image_info (filename, class_name, confidence_score, image_data) VALUES (%s, %s, %s, %s)"
      values = (filename, class_name, confidence_score, image_data)

      # Execute the query
      cursor.execute(query, values)

      # Commit the changes
      connection.commit()

  except Exception as e:
    print(f"Error storing image info: {e}")

  finally:
    # Close the connection if it is not None
    if connection is not None:
      connection.close()
@app.route("/image", methods=["GET", "POST"])
def upload_and_predict():
    if request.method == "POST":
        try:
            if 'file' not in request.files:
                return render_template("produit.html", error="No file part"), 400

            file = request.files['file']

            if file.filename == '':
                return render_template("produit.html", error="No selected file"), 400

            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                class_name, confidence_score = predict_class(file_path)

                # Get three similar images based on the predicted class
                similar_image_urls = get_extra_images(class_name)

                return render_template("produit.html", filename=filename, class_name=class_name,
                                       confidence_score=confidence_score, similar_image_urls=similar_image_urls)

        except Exception as e:
            print(f"Error processing request: {e}")
            return render_template("produit.html", error="Internal Server Error"), 500

    return render_template("produit.html", error=None, filename=None, class_name=None, confidence_score=None)
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

def save_plot_to_html(fig):
    buffer = BytesIO()
    fig.write_html(buffer, full_html=False)
    html_code = buffer.getvalue().decode()
    return html_code

def soundex_match(input_str, choices):
    best_match = None
    highest_ratio = 0

    for choice in choices:
        ratio = fuzz.soundex(input_str.lower()), fuzz.soundex(choice.lower())
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = choice

    return best_match

def generate_correction_mapping(df):
    geolocator = Nominatim(user_agent="geo_locator")
    country_list = [country.name.lower() for country in pycountry.countries]

    def correct_name(x):
        matched_country = soundex_match(x, country_list)
        return matched_country if matched_country else x

    df['Country'] = df['Country'].apply(lambda x: correct_name(x) if geolocator.geocode(x) is None else x)
    return df

def count_occurrences(file_path):
    df = pd.read_excel(file_path)
    occurrences = df['Country'].value_counts().reset_index()
    occurrences.columns = ['Country', 'Occurrences']
    return occurrences


def plot_country_map(df):
  df['Occurrences'] = df.groupby('Country')['Country'].transform('count')

  # Visualisation du nombre d'occurrences par pays sur une carte
  fig = px.choropleth(df, locations="Country", locationmode="country names", color="Occurrences",
                      title="Occurrences par pays",
                      hover_data={"Country": True, "Occurrences": True})

  # Customize the hover template using customdata
  fig.update_traces(hovertemplate='L\'occurrence du %{location} est %{customdata[1]}')

  return fig


def plot_positive_ratings(df):
    positive_ratings = df[df['Rating'] > 3]
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Rating', data=positive_ratings)
    plt.title("Nombre de Ratings Positifs (Rating > 3)")
    plt.savefig('static/positive_ratings.png')
    plt.close()

def plot_genre_ratings(df):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Gender', y='Rating', data=df)
    plt.title("Distribution du Rating par Genre")
    plt.savefig('static/genre_ratings.png')
    plt.close()

def plot_comments_vs_ratings(df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Rating', y='Comment', data=df)
    plt.title("Commentaires par rapport au Rating")
    plt.savefig('static/comments_vs_ratings.png')
    plt.close()

def plot_price_vs_ratings(df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='ProductPrice', y='Rating', data=df)
    plt.title("Rating par rapport au Prix du Produit")
    plt.savefig('static/price_vs_ratings.png')
    plt.close()

@app.route('/')
def index():
    return render_template('preparation.html')

@app.route('/analyze')
def analyze():
    file_path = "templates/analyse.xlsx"
    df = load_data(file_path)

    plot_positive_ratings(df)
    plt_genre_ratings = plot_genre_ratings(df)
    plt_comments_vs_ratings = plot_comments_vs_ratings(df)
    plt_price_vs_ratings = plot_price_vs_ratings(df)

    plt_positive_ratings_html = '/static/positive_ratings.png'
    plt_genre_ratings_html = '/static/genre_ratings.png'
    plt_comments_vs_ratings_html = '/static/comments_vs_ratings.png'
    plt_price_vs_ratings_html = '/static/price_vs_ratings.png'

    country_map_data = plot_country_map(df)
    occurrences = count_occurrences(file_path)

    return render_template('analyze.html',
                           plt_positive_ratings_html=plt_positive_ratings_html,
                           plt_genre_ratings_html=plt_genre_ratings_html,
                           plt_comments_vs_ratings_html=plt_comments_vs_ratings_html,
                           plt_price_vs_ratings_html=plt_price_vs_ratings_html,
                           country_map_data=country_map_data,
                           occurrences=occurrences)  # Ajoutez cette ligne pour transmettre 'occurrences'


if __name__ == '__main__':
    app.run(debug=True, port=5001)
