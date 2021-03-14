
from typing import Dict, Text

import os
import pprint
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

ratings = tfds.load(
  "movielens/100k-ratings",
  split="train",
)

ratings = (ratings
  .map(lambda x: {
    "user_id": x["user_id"],
    "movie_title": x["movie_title"],
  })
  .cache(tempfile.NamedTemporaryFile().name)
)

movies = tfds.load("movielens/100k-movies", split="train")
movies = (movies
  .map(lambda x: x["movie_title"])
  .cache(tempfile.NamedTemporaryFile().name)
)

user_ids = ratings.map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(user_ids.batch(1000))))

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

class MovielensModel(tfrs.Model):

  def __init__(self):
    super().__init__()

    embedding_dimension = 32

    self.movie_model = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_titles,
        mask_token=None,
      ),
      tf.keras.layers.Embedding(
        len(unique_movie_titles) + 1,
        embedding_dimension,
      ),
    ])

    self.user_model = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids,
        mask_token=None,
      ),
      tf.keras.layers.Embedding(
        len(unique_user_ids) + 1,
        embedding_dimension,
      ),
    ])

    self.task = tfrs.tasks.Retrieval(
      metrics=tfrs.metrics.FactorizedTopK(
        candidates=movies.batch(128).cache().map(self.movie_model)
      )
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    user_embeddings = self.user_model(features["user_id"])
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    return self.task(
      user_embeddings,
      positive_movie_embeddings,
      compute_metrics=not training,
    )

model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

model.fit(train.batch(8192), epochs=3)

model.evaluate(test.batch(8192), return_dict=True)

'''
brute_force = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
brute_force.index(movies.batch(128).map(model.movie_model), movies)

_, titles = brute_force(np.array(["42"]), k=3)

print(f"Top recommendations: {titles[0]}")
'''
'''
scann = tfrs.layers.factorized_top_k.ScaNN(num_reordering_candidates=1000)
scann.index(movies.batch(128).map(model.movie_model), movies)

_, titles = scann(model.user_model(np.array(["42"])), k=3)

print(f"Top recommendations: {titles[0]}")
'''

scann = tfrs.layers.factorized_top_k.ScaNN(
  model.user_model,
  num_reordering_candidates=1000,
)
scann.index(movies.batch(128).map(model.movie_model), movies)

_ = scann(np.array(["42"]))

with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")
  scann.save(
    path,
    options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]),
  )

  loaded = tf.keras.models.load_model(path)
  _, titles = loaded(tf.constant(["42"]))
  print(f"Top recommendations: {titles[0][:3]}")
