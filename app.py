from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)

# Custom Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )

        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )

        self.u = self.add_weight(
            name='context_vector',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(
            tf.tensordot(x, self.W, axes=[2, 0]) + self.b
        )

        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=[2, 0]),
            axis=1
        )

        context_vector = tf.reduce_sum(
            attention_weights[..., tf.newaxis] * x,
            axis=1
        )

        return context_vector

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Load trained model
model = load_model(
    "disease_prediction_model.h5",
    custom_objects={'AttentionLayer': AttentionLayer}
)

# Load preprocessing objects
with open("preprocessing.pkl", "rb") as f:
    preprocessing = pickle.load(f)

tokenizer = preprocessing["tokenizer"]
label_encoder = preprocessing["label_encoder"]


# Text cleaning function
def clean_text(text):
    text = text.lower()

    # Remove punctuation/special chars
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Disease prediction function
def predict_disease(symptoms):
    symptoms_cleaned = clean_text(symptoms)

    seq = tokenizer.texts_to_sequences([symptoms_cleaned])

    pad = pad_sequences(seq, maxlen=150)

    pred = model.predict(pad, verbose=0)[0]

    predicted_index = np.argmax(pred)

    predicted_label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = float(pred[predicted_index] * 100)

    return predicted_label, confidence


# Home route
@app.route('/')
def home():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    symptoms = request.form.get('symptoms', '')

    # Empty input handling
    if symptoms.strip() == "":
        return render_template(
            'index.html',
            prediction="Please enter symptoms",
            confidence=0,
            symptoms=""
        )

    try:
        prediction, confidence = predict_disease(symptoms)

        return render_template(
            'index.html',
            prediction=prediction,
            confidence=confidence,
            symptoms=symptoms
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction=f"Error: {str(e)}",
            confidence=0,
            symptoms=symptoms
        )


# Run Flask app
# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))

#     app.run(
#         host='0.0.0.0',
#         port=port
#     )