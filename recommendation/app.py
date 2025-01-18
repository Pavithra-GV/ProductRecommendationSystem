from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Set secret key for session

# Load Dataset
file_path = 'recommendation/SettylDataset.csv'  # Update with your actual path
dataset = pd.read_csv(file_path)

# Preprocess dataset
dataset['BrowsingHistory'] = dataset['BrowsingHistory'].str.replace('|', ' ')
dataset['RatingsGiven'] = dataset['RatingsGiven'] / dataset['RatingsGiven'].max()
dataset['ClickThroughRate'] = dataset['ClickThroughRate'] / dataset['ClickThroughRate'].max()
dataset['Category'] = dataset['Category'].fillna('')
dataset['Tags'] = dataset['Tags'].fillna('')
dataset['Specifications'] = dataset['Specifications'].fillna('')
dataset['Description'] = dataset['Description'].fillna('')
dataset['ProductMetadata'] = (dataset['Category'] + " " + dataset['Tags'] + " " +
                              dataset['Specifications'] + " " + dataset['Description']).str.lower()

# Initialize TF-IDF Vectorizer and compute cosine similarity for content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(dataset['ProductMetadata'])
content_similarity = cosine_similarity(tfidf_matrix)

# Create user-product interaction matrix (Collaborative Filtering)
user_product_matrix = pd.pivot_table(dataset, index='UserID', columns='ProductName', values='RatingsGiven', fill_value=0)

# Compute user similarity using cosine similarity for collaborative filtering
user_similarity = cosine_similarity(user_product_matrix)
user_similarity[np.isnan(user_similarity)] = 0  # Handle any NaN values

# Collaborative Filtering (CF) function
def predict_cf_scores(user_id):
    if user_id not in user_product_matrix.index:
        return np.zeros(user_product_matrix.shape[1])
    user_idx = user_product_matrix.index.get_loc(user_id)
    cf_scores = user_similarity[user_idx].dot(user_product_matrix) / np.array([np.abs(user_similarity[user_idx]).sum()])
    return cf_scores

# Content-Based Filtering (CBF) function
def predict_cbf_scores(user_id):
    if user_id not in user_product_matrix.index:
        return np.zeros(len(user_product_matrix.columns))
    user_products = user_product_matrix.loc[user_id]
    interacted_products_idx = user_products[user_products > 0].index
    product_to_idx = {product: idx for idx, product in enumerate(dataset['ProductName'])}
    interacted_indices = [product_to_idx[product] for product in interacted_products_idx if product in product_to_idx]
    if len(interacted_indices) == 0:
        return np.zeros(len(user_product_matrix.columns))
    cbf_scores_raw = content_similarity[interacted_indices].mean(axis=0)
    cbf_scores = np.zeros(len(user_product_matrix.columns))
    for i, product in enumerate(user_product_matrix.columns):
        if product in product_to_idx:
            cbf_scores[i] = cbf_scores_raw[product_to_idx[product]]
    return cbf_scores

# Hybrid recommendation function
def hybrid_recommendations(user_id, alpha=0.5, top_n=5):
    cf_scores = predict_cf_scores(user_id)
    cbf_scores = predict_cbf_scores(user_id)
    viewed_products = session.get('viewed_products', [])
    if viewed_products:
        for idx, product in enumerate(user_product_matrix.columns):
            if product in viewed_products:
                cf_scores[idx] *= 1.2
                cbf_scores[idx] *= 1.2
    final_scores = alpha * cf_scores + (1 - alpha) * cbf_scores
    recommended_product_indices = np.argsort(final_scores)[::-1][:top_n]
    recommended_products = user_product_matrix.columns[recommended_product_indices]
    return recommended_products

# Real-time recommendation function
def get_real_time_recommendations(viewed_products):
    product_to_idx = {product: idx for idx, product in enumerate(dataset['ProductName'])}
    viewed_indices = [product_to_idx[product] for product in viewed_products if product in product_to_idx]
    if len(viewed_indices) == 0:
        return []
    recommendations = content_similarity[viewed_indices].mean(axis=0).argsort()[::-1]
    recommended_products = [dataset['ProductName'][idx] for idx in recommendations[:5] if idx not in viewed_indices]
    return recommended_products

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "")
    if query:
        filtered_products = dataset[dataset['ProductName'].str.contains(query, case=False, na=False)]
        suggestions = filtered_products['ProductName'].tolist()
        return jsonify({"suggestions": suggestions})
    return jsonify({"suggestions": []})

@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = []
    error = None
    if request.method == "POST":
        user_id = request.form["user_id"]
        try:
            recommendation = list(hybrid_recommendations(user_id))
            if not recommendation:
                error = "Sorry, we couldn't find any recommendations based on your activity."
        except ValueError as e:
            error = f"Error processing your request: {str(e)}"
        except Exception as e:
            error = "An unexpected error occurred. Please try again later."
    return render_template("index.html", recommendation=recommendation, error=error)

@app.before_request
def track_user_activity():
    if 'viewed_products' not in session:
        session['viewed_products'] = []

@app.route("/view_product/<product_id>")
def view_product(product_id):
    if product_id not in session['viewed_products']:
        session['viewed_products'].append(product_id)
    recommendations = get_real_time_recommendations(session['viewed_products'])
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)
