
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

ratings = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings.map(lambda x: {
  "movie_title": x["movie_title"],
  "user_id": x["user_id"],
  "user_rating": x["user_rating"],
})

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

print("***************unique")
unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

print("***************定义 RankingModel")
class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    embedding_dimension = 32

    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids,
        mask_token=None,
      ),
      tf.keras.layers.Embedding(
        len(unique_user_ids) + 1,
        embedding_dimension,
      ),
    ])

    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_titles,
        mask_token=None,
      ),
      tf.keras.layers.Embedding(
        len(unique_movie_titles) + 1,
        embedding_dimension,
      )
    ])

    self.ratings = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(1),
  ])

  def call(self, inputs):
    user_id, movie_title = inputs

    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

RankingModel()((["42"], ["One Flew Over the Cuckoo's Nest (1975)"]))

task = tfrs.tasks.Ranking(
  loss = tf.keras.losses.MeanSquaredError(),
  metrics = [tf.keras.metrics.RootMeanSquaredError()]
)

class MovielensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics = [tf.keras.metrics.RootMeanSquaredError()],
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    rating_predictions = self.ranking_model(
      (features["user_id"], features["movie_title"])
    )

    return self.task(
      labels=features["user_rating"],
      predictions=rating_predictions,
    )

model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

print("*********************fit")
model.fit(cached_train, epochs=3)

print("*********************evaluate")
model.evaluate(cached_test, return_dict=True)
