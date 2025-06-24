from keras.models import load_model
model = load_model('music_genre_model.keras')
model.summary()
json_config = model.to_json()
with open("model_config.json", "w") as json_file:
    json_file.write(json_config)
