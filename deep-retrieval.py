
import os
import tempfile

import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

plt.style.use('seaborn-whitegrid')

ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
  "movie_title": x["movie_title"],
  "user_id": x["user_id"],
  "timestamp": x["timestamp"],
})
movies = movies.map(lambda x: x["movie_title"])

timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

timestamp_buckets = np.linspace(
  min_timestamp,
  max_timestamp,
  num=1000,
)

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
  lambda x: x["user_id"]
))))

print("class UserModel")
class UserModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    print("UserModel-user_embedding")
    self.user_embedding = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids,
        mask_token=None,
      ),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
    ])
    print("UserModel-timestamp_embedding")
    self.timestamp_embedding = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
      tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
    ])
    print("UserModel-normalized_timestamp")
    self.normalized_timestamp = tf.keras.layers.experimental.preprocessing.Normalization()

    print("UserModel-normalized_timestamp-adapt")
    self.normalized_timestamp.adapt(timestamps)

  def call(self, inputs):
    return tf.concat([
      self.user_embedding(inputs["user_id"]),
      self.timestamp_embedding(inputs["timestamp"]),
      self.normalized_timestamp(inputs["timestamp"]),
    ], axis=1)

print("class QueryModel")
class QueryModel(tf.keras.Model):

  def __init__(self, layer_sizes):
    super().__init__()

    self.embedding_model = UserModel()

    print("QueryModel-dense_layers")
    self.dense_layers = tf.keras.Sequential()

    print("QueryModel-dense_layers-add-relu")
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    print("QueryModel-dense_layers-add")
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)

print("class QMovieModel")
class MovieModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    max_tokens = 10_000

    print("MovieModel-title_embedding")
    self.title_embedding = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_titles,
        mask_token=None,
      ),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
    ])

    print("MovieModel-title_vectorizer")
    self.title_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=max_tokens)

    print("MovieModel-title_text_embedding")
    self.title_text_embedding = tf.keras.Sequential([
      self.title_vectorizer,
      tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      tf.keras.layers.GlobalAveragePooling1D(),
    ])

    print("MovieModel-title_vectorizer-adapt")
    self.title_vectorizer.adapt(movies)

  def call(self, titles):
    return tf.concat([
      self.title_embedding(titles),
      self.title_text_embedding(titles),
    ], axis=1)

print("class CandidateModel")
class CandidateModel(tf.keras.Model):

  def __init__(self, layer_sizes):
    super().__init__()

    self.embedding_model = MovieModel()

    print("CandidateModel-dense_layers")
    self.dense_layers = tf.keras.Sequential()

    print("CandidateModel-dense_layers-add-relu")
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    print("CandidateModel-dense_layers-add")
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))

  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)

print("class MovielensModel")
class MovielensModel(tfrs.models.Model):

  def __init__(self, layer_sizes):
    super().__init__()
    self.query_model = QueryModel(layer_sizes)
    print("MovielensModel-candidate_model")
    self.candidate_model = CandidateModel(layer_sizes)
    print("MovielensModel-task")
    self.task = tfrs.tasks.Retrieval(
      metrics=tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).map(self.candidate_model),
      ),
    )

  def compute_loss(self, features, training=False):

    print("MovielensModel-query_embeddings")
    query_embeddings = self.query_model({
      "user_id": features["user_id"],
      "timestamp": features["timestamp"],
    })
    print("MovielensModel-movie_embeddings")
    movie_embeddings = self.candidate_model(features["movie_title"])

    return self.task(
      query_embeddings,
      movie_embeddings,
      compute_metrics=not training
    )

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()

#num_epochs = 300
num_epochs = 100

print("实例model")
model = MovielensModel([32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

print("model fit")
one_layer_history = model.fit(
  cached_train,
  validation_data=cached_test,
  validation_freq=5,
  epochs=num_epochs,
  verbose=0,
)
print("one_layer_history")
accuracy = one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}")

print("实例model")
model = MovielensModel([64, 32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

print("model fit")
two_layer_history = model.fit(
  cached_train,
  validation_data=cached_test,
  validation_freq=5,
  epochs=num_epochs,
  verbose=0,
)
accuracy = two_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy-two-layer: {accuracy:.2f}")

print("plot-start")
num_validation_runs = len(one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"])
epochs = [(x + 1) * 5 for x in range(num_validation_runs)]

plt.plot(epochs, one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="1 layer")
plt.plot(epochs, two_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="2 layer")
plt.title("Accuracy vs epoch")
plt.xlabel("epoch")
plt.ylabel("Top-100 accuracy")
plt.legend()
print("plot")
