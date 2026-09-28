# Book Recommendation Engine using KNN

A book recommendation system built with scikit-learn's `NearestNeighbors` on the Book-Crossings dataset. Given a book title, it returns the five most similar books along with their distances. This project is part of the freeCodeCamp Machine Learning with Python certification.

## Overview

- Dataset: Book-Crossings, about 1.1 million ratings (1-10) of 270,000 books by 90,000 users
- Users with fewer than 200 ratings and books with fewer than 100 ratings are removed for statistical significance
- Ratings are arranged in a book-by-user matrix and stored as a sparse CSR matrix
- Similarity is measured with cosine distance using a brute-force nearest neighbors search

## How It Works

1. Load the books and ratings data into pandas DataFrames.
2. Filter out users and books with too few ratings.
3. Merge ratings with book titles and pivot into a title-by-user matrix, filling missing values with 0.
4. Fit a `NearestNeighbors` model (`metric='cosine'`, `algorithm='brute'`).
5. `get_recommends(title)` finds the 6 nearest neighbors (the first is the book itself), drops the book itself, and returns the other five as `[title, [[book, distance], ...]]`.

## Example

```python
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
```

Returns the book title followed by five similar books, each paired with its distance from the input book.

## Results

The model passes the freeCodeCamp test for the input book "Where the Heart Is (Oprah's Book Club (Paperback))".

## Getting Started

### Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/fcc_book_recommendation_knn_completed.ipynb)

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook fcc_book_recommendation_knn_completed.ipynb
```

The notebook downloads the dataset with `!wget` and `!unzip`, which need a Unix shell. On Windows, use Colab or WSL.

## Project Structure

```
.
├── fcc_book_recommendation_knn_completed.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── images/
```

## Technologies

Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib

## Acknowledgments

Dataset and challenge by [freeCodeCamp](https://www.freecodecamp.org).
