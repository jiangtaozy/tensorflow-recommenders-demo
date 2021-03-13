
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

# 获取评分列表
pprint.pprint("加载数据***********************")
ratings = tfds.load("movielens/100k-ratings", split="train")
# 获取电影列表
movies = tfds.load("movielens/100k-movies", split="train")

# 打印一项评分
pprint.pprint("打印***********************")
for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

# 打印一项电影
for x in movies.take(1).as_numpy_iterator():
  pprint.pprint(x)

# 只保留 user_id 和 movie_title
pprint.pprint("处理数据***********************")
ratings = ratings.map(lambda x: {
  "movie_title": x["movie_title"],
  "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])

# 打印一项评分
pprint.pprint("打印***********************")
for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

# 打印一项电影
for x in movies.take(1).as_numpy_iterator():
  pprint.pprint(x)

# 80%训练，20%测试
pprint.pprint("分组***********************")
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# 去重
pprint.pprint("去重***********************")
movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

pprint.pprint(unique_movie_titles[:10])

# 确定维度
embedding_dimension = 32

# 用户模型
pprint.pprint("创建模型***********************")
user_model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=unique_user_ids,
    mask_token=None,
  ),
  tf.keras.layers.Embedding(
      len(unique_user_ids) + 1,
      embedding_dimension,
  ),
])

# 电影模型
movie_model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=unique_movie_titles,
    mask_token=None,
  ),
  tf.keras.layers.Embedding(
    len(unique_movie_titles) + 1,
    embedding_dimension,
  ),
])

# Metrics 指标
metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(movie_model)
)

# Loss 损失
task = tfrs.tasks.Retrieval(
  metrics=metrics
)

# 全模型
class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    user_embeddings = self.user_model(features["user_id"])
    positive_movie_embeddings = self.movie_model(features["movie_title"])
    return self.task(user_embeddings, positive_movie_embeddings)

# 实例化模型
pprint.pprint("实例化模型***********************")
model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# 缓存
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# 训练模型
pprint.pprint("训练模型***********************")
model.fit(cached_train, epochs=3)

# 评估模型
pprint.pprint("评估模型***********************")
model.evaluate(cached_test, return_dict=True)

# 预测
pprint.pprint("预测***********************")
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index(movies.batch(100).map(model.movie_model), movies)
_, titles = index(tf.constant(["42"]))
print(f"推荐用户42: {titles[0, :3]}")

# 保存index
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")

  # 保存index
  index.save(path)

  loaded = tf.keras.models.load_model(path)

  scores, titles = loaded(["42"])

  print(f"推荐*****************：{titles[0][:3]}")

# 预测-处理千万候选项
print("处理千万候选项*********************")
scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann_index.index(movies.batch(100).map(model.movie_model), movies)

# 推荐
_, titles = scann_index(tf.constant(["42"]))
print(f"推荐：*******************42：{titles[0, :3]}")

# 保存index
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")
  scann_index.save(
    path,
    options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
  )
  loaded = tf.keras.models.load_model(path)
  scores, titles = loaded(["42"])
  print(f"推荐：********************42: {titles[0][:3]}")
