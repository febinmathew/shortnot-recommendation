import os
import numpy as np
# pip install numpy

import pandas as pd 
# pip install pandas

import tensorflow as tf
# pip install tensorflow

# import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
# pip install tensorflow_recommenders

from tensorflow import keras
import datetime

# model = tf.keras.models.load_model('model_weights/weights')
import requests
import json


# # from flask import Flask
# from flask import request, Flask, jsonify
# app = Flask(__name__)

startFreshTrain=True
isLocalServer=True

def getUserVideoLikesWithVideoDescription():
    fetchUrl="http://localhost/shortnot/api/admin/getAllVideoWatchWithLikeAndVideoData"
    headers = {"Api-Key": "156c4675-9608-4591-1111-00000","User-Id":"25","Auth-Token":"yOy7KJMyCzAwYlIndsEF42G_pkITf88NR_meJF7enror2gMenKU5Pf27fNX2OG_0vpr0muNDrRDS6RcJhcAEiQ=="}
            
    x = requests.post(fetchUrl,headers=headers, data = {})
    return json.loads( x.text )["msg"]
    # return [{'id': 8, 'user_id': 2, 'fb_id': '', 'description': '#seavibes #funtime #sunset', 'video': 'https://firebasestorage.googleapis.com/v0/b/toktok-db796.appspot.com/o/post_data%2F2%2F2022_8_18%2F62fe371983d4b2.mp4?alt=media&token=e57b52f2-9899-43fa-8744-f29291bcb5b4', 'thum': 'https://firebasestorage.googleapis.com/v0/b/toktok-db796.appspot.com/o/post_data%2F2%2F2022_8_18%2F62fe371983d4b2.png?alt=media&token=3b9ac204-efec-4e3f-9cf5-fa3d66e80eea', 'gif': 'https://firebasestorage.googleapis.com/v0/b/toktok-db796.appspot.com/o/post_data%2F2%2F2022_8_18%2F62fe371983d4b2.gif?alt=media&token=c9d518a0-a7f9-42ac-a002-df72ca33dcb7', 'view': 123, 'section': '0', 'sound_id': 0, 'privacy_type': 'public', 'allow_comments': 'true', 'allow_duet': 0, 'block': 0, 'duet_video_id': 0, 'old_video_id': 0, 'duration': 20, 'created': '2022-08-18T12:56:57+00:00', 'sound': None, 'user': {'id': 2, 'first_name': 'Febin', 'last_name': 'Mathew', 'gender': '', 'bio': '', 'website': '', 'dob': '1983-08-06T00:00:00+00:00', 'social_id': '', 'email': '', 'phone': '+919633323461', 'password': '', 'profile_pic': 'app/webroot/uploads/images/6301bfe4d9003.png', 'role': 'user', 'username': 'febz', 'social': '', 'device_token': '', 'token': '', 'active': 1, 'lat': '', 'long': '', 'online': 0, 'verified': 0, 'auth_token': '', 'version': '2.2', 'device': 'android', 'ip': '103.154.54.172', 'city': 'kozhikode', 'country': 'india', 'city_id': 132600, 'state_id': 4028, 'country_id': 101, 'fb_id': '', 'created': '2022-08-06T11:31:58+00:00', 'push_notification': {'id': 2, 'likes': 1, 'comments': 1, 'new_followers': 1, 'mentions': 1, 'direct_messages': 1, 'video_updates': 1}, 'privacy_setting': {'id': 2, 'videos_download': 1, 'direct_message': 'everyone', 'duet': 'everyone', 'liked_videos': 'me', 'video_comment': 'everyone'}}}]

def trainVideoRanking():
    global startFreshTrain
    videoCollection = getUserVideoLikesWithVideoDescription()
    
    rating_data = pd.DataFrame([t for t in videoCollection])
    # rating_data['user_id'] = rating_data['user_id'].astype(int)
    # rating_data['video_id'] = rating_data['video_id'].astype(int)
    # rating_data['video_rating'] = rating_data['video_rating'].astype(float)

    # print(rating_data['video_rating'].values)
    
    # tf.cast(rating_data['user_id'].values.reshape(-1,1), tf.int64)
    
    # tf.cast(rating_data['video_id'].values.reshape(-1,1), tf.int64)
    
    # tf.cast(rating_data['video_rating'].values.reshape(-1,1), tf.float32)
    # exit()

    # dataset = tf.data.Dataset.from_tensor_slices((tf.cast(rating_data['user_id'].values.reshape(-1,1), tf.int64),tf.cast(rating_data['video_id'].values.reshape(-1,1), tf.int64), tf.cast(rating_data['video_description'].values.reshape(-1,1), tf.string), 
    # tf.cast(rating_data['video_rating'].values.reshape(-1,1),tf.float32)))


    rating_data['user_id'] = rating_data.user_id.astype(np.int64)
    rating_data['video_id'] = rating_data.video_id.astype(np.int64)
    rating_data['video_description'] = rating_data.video_description.astype(np.str)
    rating_data['video_rating'] = rating_data.video_rating.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(rating_data['user_id'].values.reshape(-1,1), tf.int64),tf.cast(rating_data['video_id'].values.reshape(-1,1), tf.int64),    tf.cast(rating_data['video_description'].values.reshape(-1,1), tf.string),
    tf.cast(rating_data['video_rating'].values.reshape(-1,1),tf.float32)))

    @tf.function
    def rename(x0,x1,x2,x3):
        y = {}
        y["user_id"] = x0
        y['video_id'] = x1
        y['video_description'] = x2
        y['video_rating'] = x3
        return y

    dataset = dataset.map(rename)

    video_descriptions = rating_data.video_description.values
    users = rating_data.user_id.values
    videos = rating_data.video_id.values

    unique_video_descriptions = np.unique(list(video_descriptions))
    unique_user_ids = np.unique(list(users))
    unique_video_ids = np.unique(list(videos))
    # print(len(unique_video_ids))
    # exit()


    class RankingModel(tf.keras.Model):

        def __init__(self):
            super().__init__()
            embedding_dimension = 32

            # Compute embeddings for users.
            self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.IntegerLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
            ])

            # Compute embeddings for books.
            self.video_description_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_video_descriptions, mask_token=None),
            tf.keras.layers.Embedding(len(unique_video_descriptions) + 1, embedding_dimension)
            ])

            # Compute embeddings for books.
            self.video_id_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.IntegerLookup(
                vocabulary=unique_video_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_video_ids) + 1, embedding_dimension)
            ])

            # Compute predictions.
            self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

        def call(self, inputs):
            
            user_id, video_id,video_description = inputs
            # print("user_id<<<<<<<<<<<<<<MAAAAAAAAAAAA",user_id,video_id,video_description)
            # exit()
            user_embedding = self.user_embeddings(user_id)
            video_description_embedding = self.video_description_embeddings(video_description)
            video_id_embeddings = self.video_id_embeddings(video_id)
            # print(book_embedding)
            # exit()

            return self.ratings(tf.concat([user_embedding,video_description_embedding], axis=1))
            # return self.ratings(tf.concat([user_embedding,video_description_embedding,video_id_embeddings], axis=1))

        # model = RankingModel()    


    class VideoModel(tfrs.models.Model):

        def __init__(self):
            super().__init__()
            self.ranking_model: tf.keras.Model = RankingModel()
            self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                loss = tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )
        def call(self, features: dict[str, tf.Tensor]) -> tf.Tensor:

            # print(features.user_id)
            user_id=features['user_id']
            video_id=features['video_id']
            video_description=features['video_description']
            
            
            # print(user_id)
            # exit()
            # rating_predictions = self.ranking_model((user_id,video_id, video_description))
            
            return self.ranking_model(
                (user_id, video_id,video_description))

        def compute_loss(self, features, training=False) -> tf.Tensor:
            video_rating=features['video_rating']
            rating_predictions = self(features)
            return self.task(labels=video_rating, predictions=rating_predictions)

    
    model = VideoModel()

    if(not startFreshTrain):
        model = tf.saved_model.load('weights')
    
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    # Cache the dataset 
    cached_train = dataset.cache()
    # cached_test  = dataset.cache()
    # Tensorboard 
    logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # Training 
    model.fit(cached_train, epochs=2,
                        verbose=1, callbacks=[tensorboard_callback])
    
    

    # model.save('weights')
    tf.saved_model.save(model, 'weights')
    uploadModelToSever()
    # exit()
    # model.evaluate(cached_test , return_dict=True)

def uploadModelToSever():
    if (isLocalServer):
        url='http://127.0.0.1:43657/uploadModelFiles'
    else:    
        url='https://shortnot.returnbyte.com:43657/uploadModelFiles'


    filePath='weights/saved_model.pb'
    with open(filePath, 'rb') as f:
        files = {'upload_file': f.read()}
    values = {'file_dir': filePath, 'name': 'saved_model.pb'}    
    r = requests.post(url, files=files, data=values)


    filePath='weights/variables/variables.data-00000-of-00001'
    with open(filePath, 'rb') as f:
        files = {'upload_file': f.read()}
    values = {'file_dir': filePath, 'name': 'variables.data-00000-of-00001'}    
    r = requests.post(url, files=files, data=values)

    filePath='weights/variables/variables.index'
    with open(filePath, 'rb') as f:
        files = {'upload_file': f.read()}
    values = {'file_dir': filePath, 'name': 'variables.index'}    
    r = requests.post(url, files=files, data=values)

# uploadModelToSever()
trainVideoRanking()