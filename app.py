import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

IMG_SIZE = (150, 150)
class_names = ['NORMAL', 'PNEUMONIA']

# ── Rebuild the EXACT same architecture as training ──
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
normalization_layer = tf.keras.layers.Rescaling(1./255)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    data_augmentation,
    normalization_layer,
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Build the model by calling it once, then load trained weights
model.build(input_shape=(None, 150, 150, 3))
model.load_weights('pneumonia_cnn.weights.h5')

# ── Find last conv layer automatically ──────────────
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

# ── Grad-CAM functions ───────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x, training=False)
        if layer.name == last_conv_layer_name:
            conv_output = x
    grad_model = tf.keras.models.Model(inputs, [conv_output, x])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img_uint8 = np.uint8(img)
    return cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)

# ── Streamlit UI ─────────────────────────────────────
st.title("🩺 Pneumonia Detection from Chest X-Rays")
st.write("Chest X-ray upload karo — model bataega Normal hai ya Pneumonia, saath mein heatmap bhi dikhega ke model ne kaha dekha.")

uploaded_file = st.file_uploader("X-ray image upload karo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype('float32')
    input_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Analyzing X-ray...'):
        prediction = model.predict(input_array)[0][0]
        pred_class = 1 if prediction > 0.5 else 0
        confidence = prediction if pred_class == 1 else 1 - prediction

        heatmap = make_gradcam_heatmap(input_array, model, last_conv_layer_name)
        overlayed_img = overlay_gradcam(img_array, heatmap)

    result_text = "🫁 PNEUMONIA DETECTED" if pred_class == 1 else "✅ NORMAL"
    st.subheader(result_text)
    st.write(f"Confidence: {confidence*100:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_array.astype('uint8'), caption="Original X-Ray", use_container_width=True)
    with col2:
        st.image(overlayed_img, caption="Grad-CAM Heatmap", use_container_width=True)