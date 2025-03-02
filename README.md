# Product Recommendation System App

## 1. Problem

**Challenge:**  
Modern e-commerce platforms and online retail stores offer a vast number of products, which often overwhelms users. Navigating this extensive catalog to find items that match individual tastes can be time-consuming and frustrating.

**Key Issues:**
- **Information Overload:** Users may feel lost among the countless available products.
- **Generic Recommendations:** A one-size-fits-all approach does not cater to individual preferences.
- **User Engagement:** Poor product discovery can result in lower user satisfaction and engagement, ultimately affecting sales.

---

## 2. Why We Need This App

**Enhanced User Experience:**  
Personalized recommendations help users quickly discover products that match their interests, leading to a smoother shopping experience.

**Increased Engagement and Sales:**  
By showing relevant products based on user behavior and preferences, the app encourages more interaction and can boost conversion rates.

**Competitive Advantage:**  
E-commerce platforms that offer tailored product suggestions can stand out in a crowded market, providing a unique value proposition to users.

**Efficient Browsing:**  
Dynamic search suggestions and real-time recommendations help streamline the shopping process, reducing the time spent searching for the right product.

---

## 3. About the App and What It Does

This Flask-based app is designed to provide personalized product recommendations using a hybrid recommendation approach that blends:

- **Collaborative Filtering (CF):**  
  Uses historical user ratings and interactions to find patterns among users and recommend products that similar users liked.

- **Content-Based Filtering (CBF):**  
  Leverages product metadata (such as category, tags, specifications, and descriptions) using TF-IDF vectorization to find products similar in content to those the user has shown interest in.

**Key Functionalities:**

- **Dynamic Search Suggestions:**  
  As the user types in the search bar, the app fetches and displays real-time suggestions based on product names.

- **User Activity Tracking:**  
  The app maintains a session-based record of products viewed by the user, which it uses to adjust recommendations.

- **Hybrid Recommendations:**  
  Combines the outputs from both CF and CBF techniques. It even boosts scores for products that the user has already interacted with, providing a more tailored recommendation list.

- **Real-Time Updates:**  
  Recommendations update instantly as the user interacts with the app, ensuring that suggestions remain relevant.

---

## 4. Code Implementation Part of the App

### a. `app.py`

- **Library Imports & Setup:**  
  The app imports necessary libraries such as Flask for web handling, Pandas and NumPy for data manipulation, and Scikit-learn for computing TF-IDF and cosine similarity.

- **Dataset Loading and Preprocessing:**  
  - The dataset is loaded from a CSV file.
  - Columns like `BrowsingHistory`, `RatingsGiven`, `ClickThroughRate`, and product details (`Category`, `Tags`, `Specifications`, `Description`) are processed.
  - A new column, `ProductMetadata`, is created by combining and lowering the text from multiple fields.
  
  ```python
  # Load Dataset and preprocess
  dataset = pd.read_csv(file_path)
  dataset['BrowsingHistory'] = dataset['BrowsingHistory'].str.replace('|', ' ')
  dataset['RatingsGiven'] = dataset['RatingsGiven'] / dataset['RatingsGiven'].max()
  # Filling missing values and creating a combined metadata field
  dataset['Category'] = dataset['Category'].fillna('')
  dataset['Tags'] = dataset['Tags'].fillna('')
  dataset['Specifications'] = dataset['Specifications'].fillna('')
  dataset['Description'] = dataset['Description'].fillna('')
  dataset['ProductMetadata'] = (dataset['Category'] + " " + dataset['Tags'] + " " +
                                dataset['Specifications'] + " " + dataset['Description']).str.lower()
  ```
### Content-Based Filtering (CBF):

A TF-IDF Vectorizer converts the product metadata into vectors.
Cosine similarity is computed between these vectors to assess the similarity between products.
```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(dataset['ProductMetadata'])
content_similarity = cosine_similarity(tfidf_matrix)
```

### Collaborative Filtering (CF):

A user-product interaction matrix is created using a pivot table.
Cosine similarity is used to determine similarities between users based on their ratings.
```python
user_product_matrix = pd.pivot_table(dataset, index='UserID', columns='ProductName', values='RatingsGiven', fill_value=0)
user_similarity = cosine_similarity(user_product_matrix)
user_similarity[np.isnan(user_similarity)] = 0  # Handling NaN values
```
### Hybrid Recommendation Function:
Combines CF and CBF scores and adjusts them based on products the user has already viewed.

```python
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
```

### Web Routes:

**/** Route: Handles both GET and POST requests to render the main page and display recommendations.

**/search** Route: Provides dynamic product suggestions based on user input.

**/view_product/<product_id>** Route: Updates the session with viewed products and provides real-time recommendations.

### b. **index.html** & **style.css**

#### index.html:
Contains the HTML structure for the UI.
Implements a search bar that uses JavaScript to fetch search suggestions.
Provides a form for user ID submission and displays the recommended products.

#### style.css:
Styles the application for a clean, user-friendly interface.
Defines the layout, typography, and visual aesthetics of the app components.

---

## 5. Summary
This product recommendation system app addresses the challenge of overwhelming product catalogs in e-commerce by providing personalized and dynamic recommendations. By combining collaborative and content-based filtering techniques, the app offers a hybrid approach that tailors suggestions based on both user behavior and product similarities. The implementation leverages Flask for web functionality, Pandas and NumPy for data processing, and Scikit-learn for computing similarities. With real-time search suggestions and activity tracking, the app enhances the user experience, leading to increased engagement and potential sales uplift.

Overall, this solution demonstrates a practical, scalable, and effective approach to solving the modern problem of product discovery in digital commerce.
