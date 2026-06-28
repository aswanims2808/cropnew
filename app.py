import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

import tensorflow as tf
model = tf.keras.models.load_model('crop_disease_model_best.h5', compile=False)

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

DISEASE_DB = {
    'Apple___Apple_scab': {'display':'Apple Scab','healthy':False,'severity':'Moderate','info':'A fungus causes dark, scab-like spots on leaves and fruit. Spreads in cool, wet spring weather.','steps':['Remove fallen leaves in autumn.','Prune trees for airflow.','Apply sulfur/copper fungicide in wet spring.','Choose scab-resistant varieties.']},
    'Apple___Black_rot': {'display':'Black Rot','healthy':False,'severity':'High','info':'Purple spots on leaves and rotting mummified fruit. Enters through wounds or dead wood.','steps':['Cut diseased branches in dry weather.','Remove mummified fruit.','Apply fungicide at bud break.','Avoid wounding trees.']},
    'Apple___Cedar_apple_rust': {'display':'Cedar Apple Rust','healthy':False,'severity':'Moderate','info':'Bright orange-yellow spots. Needs both apple and cedar trees to complete its life cycle.','steps':['Remove nearby cedar/juniper trees.','Apply protective fungicide in early spring.','Plant rust-resistant varieties.']},
    'Apple___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Foliage is vibrant and free of lesions or fungal spores.','steps':['Maintain standard watering.','Continue seasonal fertilization.']},
    'Blueberry___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'No visible signs of disease detected.','steps':['Maintain acidic soil pH (4.5–5.5).','Ensure consistent soil moisture.']},
    'Cherry_(including_sour)___Powdery_mildew': {'display':'Powdery Mildew','healthy':False,'severity':'Moderate','info':'White powdery fungal growth on leaves and shoots, stunting growth.','steps':['Prune crowded branches.','Apply neem oil early.','Use sulfur-based fungicides preventatively.']},
    'Cherry_(including_sour)___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'No powdery residue or spotting detected.','steps':['Maintain standard orchard care.']},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'display':'Gray Leaf Spot','healthy':False,'severity':'High','info':'Pale brown to gray rectangular lesions running parallel to leaf veins.','steps':['Rotate crops with non-host crops.','Deep-till to bury infected residue.','Apply foliar fungicide during tasseling.']},
    'Corn_(maize)___Common_rust_': {'display':'Common Rust','healthy':False,'severity':'Moderate','info':'Powdery reddish-brown pustules on both leaf surfaces.','steps':['Plant rust-resistant hybrid varieties.','Apply foliar fungicide before silking.']},
    'Corn_(maize)___Northern_Leaf_Blight': {'display':'Northern Leaf Blight','healthy':False,'severity':'High','info':'Large cigar-shaped grayish lesions reducing photosynthesis significantly.','steps':['Select resistant hybrids.','Rotate away from corn for a year.','Apply fungicides before tasseling.']},
    'Corn_(maize)___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Leaves are uniformly green with no necrotic lesions.','steps':['Maintain nitrogen levels.','Monitor for pests.']},
    'Grape___Black_rot': {'display':'Black Rot','healthy':False,'severity':'High','info':'Circular brown lesions on leaves; grapes turn into hard black mummies.','steps':['Remove all mummified fruit.','Ensure canopy sunlight penetration.','Apply protective fungicides from early shoot growth.']},
    'Grape___Esca_(Black_Measles)': {'display':'Esca (Black Measles)','healthy':False,'severity':'Critical','info':'Wood disease causing tiger-stripe discoloration and dark spotting on fruit.','steps':['Avoid pruning in wet weather.','Remove and burn infected vines.','Apply wound protectants after pruning.']},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'display':'Leaf Blight','healthy':False,'severity':'Moderate','info':'Dark brown lesions causing premature defoliation and weakened vines.','steps':['Apply copper-based fungicides.','Clear dead leaves in winter.']},
    'Grape___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Canopy is green, vigorous and free of spotted lesions.','steps':['Continue canopy management and trellising.']},
    'Orange___Haunglongbing_(Citrus_greening)': {'display':'Citrus Greening (HLB)','healthy':False,'severity':'Critical','info':'Fatal bacterial disease causing asymmetrical yellowing and bitter green fruit.','steps':['No cure; uproot and destroy infected trees.','Apply insecticides to control psyllid vector.','Buy certified disease-free nursery stock.']},
    'Peach___Bacterial_spot': {'display':'Bacterial Spot','healthy':False,'severity':'Moderate','info':'Water-soaked spots that fall out creating a shot-hole appearance on leaves.','steps':['Apply copper bactericides at leaf drop.','Maintain optimal tree nutrition.','Plant resistant cultivars.']},
    'Peach___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Leaves are solid and free of bacterial shot-holes or chlorosis.','steps':['Maintain dormant oil sprays.','Thin fruit to prevent branch breaking.']},
    'Pepper,_bell___Bacterial_spot': {'display':'Bacterial Spot','healthy':False,'severity':'Moderate','info':'Small spots causing severe leaf drop and sun-scalded fruit.','steps':['Use certified disease-free seeds.','Use drip irrigation.','Apply copper sprays with mancozeb.']},
    'Pepper,_bell___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Foliage is dark green with no bacterial lesions.','steps':['Provide consistent watering and balanced fertilizer.']},
    'Potato___Early_blight': {'display':'Early Blight','healthy':False,'severity':'Moderate','info':'Dark sunken spots with concentric target-like rings starting on older leaves.','steps':['Remove infected lower leaves promptly.','Apply chlorothalonil or copper fungicides.','Mulch around plants.','Rotate crops yearly.']},
    'Potato___Late_blight': {'display':'Late Blight','healthy':False,'severity':'Critical','info':'Aggressive water mold with dark water-soaked lesions that devastates crops rapidly.','steps':['Destroy infected plants immediately.','Apply preventative fungicides.','Ensure rapid canopy drying.']},
    'Potato___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Leaves show no signs of necrotic rings or blight.','steps':['Hill soil around plants.','Maintain even soil moisture.']},
    'Raspberry___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Canes and leaves are vigorous with no rust or blight.','steps':['Prune old canes after fruiting.','Ensure good drainage.']},
    'Soybean___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Uniform green foliage with no rust or nematode damage.','steps':['Monitor for aphids and stinkbugs during pod development.']},
    'Squash___Powdery_mildew': {'display':'Powdery Mildew','healthy':False,'severity':'Moderate','info':'White dusty fungal growth covering leaves, preventing photosynthesis.','steps':['Plant in full sun with space between vines.','Apply neem oil at first sign.','Remove severely affected leaves.']},
    'Strawberry___Leaf_scorch': {'display':'Leaf Scorch','healthy':False,'severity':'Moderate','info':'Purplish-brown blotches that dry up and look scorched on leaves.','steps':['Remove infected dead leaves.','Renew plantings every 3–4 years.','Apply fungicides in early spring.']},
    'Strawberry___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Leaves are green with no purple scorching.','steps':['Maintain straw mulch.','Ensure consistent watering.']},
    'Tomato___Bacterial_spot': {'display':'Bacterial Spot','healthy':False,'severity':'High','info':'Dark greasy spots on leaves and scabby lesions on fruit. Thrives in warm rain.','steps':['Use drip irrigation.','Apply copper + mancozeb.','Do not work in the patch when plants are wet.']},
    'Tomato___Early_blight': {'display':'Early Blight','healthy':False,'severity':'Moderate','info':'Target-like brown spots with yellow halos starting on the lowest leaves.','steps':['Prune lowest branches so none touch soil.','Mulch heavily.','Apply chlorothalonil fungicides.']},
    'Tomato___Late_blight': {'display':'Late Blight','healthy':False,'severity':'Critical','info':'Large dark water-soaked patches on leaves and stems, quickly killing the plant.','steps':['Pull and bag infected plants immediately.','Apply copper-based sprays before wet weather.']},
    'Tomato___Leaf_Mold': {'display':'Leaf Mold','healthy':False,'severity':'Moderate','info':'Pale yellow spots on upper leaf; olive-green fuzzy mold on the underside.','steps':['Maximize airflow by pruning.','Reduce greenhouse humidity.']},
    'Tomato___Septoria_leaf_spot': {'display':'Septoria Leaf Spot','healthy':False,'severity':'Moderate','info':'Small circular spots with gray centers causing rapid defoliation.','steps':['Remove infected lower leaves.','Apply fungicidal sprays every 7–10 days.']},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'display':'Spider Mites','healthy':False,'severity':'High','info':'Microscopic pests causing stippled yellow leaves covered in fine webbing.','steps':['Spray with strong water stream.','Apply insecticidal soap or neem oil.']},
    'Tomato___Target_Spot': {'display':'Target Spot','healthy':False,'severity':'Moderate','info':'Sunken target-like spots on leaves and fruit leading to crop loss.','steps':['Improve airflow.','Apply targeted fungicides.','Avoid excess nitrogen.']},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'display':'Yellow Leaf Curl Virus','healthy':False,'severity':'Critical','info':'Transmitted by whiteflies. Leaves curl upward and turn yellow at edges.','steps':['Control whiteflies with horticultural oils.','Destroy infected plants.','Use reflective mulches.']},
    'Tomato___Tomato_mosaic_virus': {'display':'Mosaic Virus','healthy':False,'severity':'High','info':'Contagious virus causing mottled mosaic pattern and stunted growth.','steps':['No cure; destroy infected plants.','Wash hands and tools with soap.','Avoid tobacco products near plants.']},
    'Tomato___healthy': {'display':'Healthy','healthy':True,'severity':'None','info':'Leaves are vibrant, unspotted, and free of curling or webbing.','steps':['Maintain consistent watering.','Feed with calcium-rich fertilizer.']}
}

print("✅ Model and database loaded.")

def make_gradcam_heatmap(img_array, model, pred_index=None):
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer('out_relu')
    conv_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[1:]:
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        preds = classifier_model(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def analyze_leaf(pil_image):
    if pil_image is None:
        return None, None, "<p>Please upload an image.</p>"

    # Preprocess Image
    img = pil_image.resize((224, 224))
    arr = np.array(img) / 255.0
    tensor = np.expand_dims(arr, axis=0).astype(np.float32)

    # Predict
    preds = model.predict(tensor, verbose=0)[0]
    top5_idx = np.argsort(preds)[-5:][::-1]
    top5_conf = preds[top5_idx] * 100
    top5_names = [CLASS_NAMES[i].replace('___', ' — ').replace('_', ' ') for i in top5_idx]

    predicted_class = CLASS_NAMES[top5_idx[0]]
    confidence = float(preds[top5_idx[0]]) * 100
    db = DISEASE_DB.get(predicted_class, {})
    is_healthy = db.get('healthy', False)
    display_name = db.get('display', predicted_class)
    crop_name = predicted_class.split('___')[0].replace('_', ' ')
    severity = db.get('severity', 'Unknown')
    info = db.get('info', '')
    steps = db.get('steps', [])

    # Grad-CAM overlay (Memory efficient, no disk saving)
    heatmap = make_gradcam_heatmap(tensor, model, pred_index=int(top5_idx[0]))
    
    # Convert PIL directly to OpenCV format
    orig_rgb = np.array(pil_image.convert('RGB')) 
    orig_resized = cv2.resize(orig_rgb, (400, 400))
    
    hmap_resized = cv2.resize(heatmap, (400, 400))
    hmap_colored = cv2.applyColorMap(np.uint8(255 * hmap_resized), cv2.COLORMAP_JET)
    hmap_colored = cv2.cvtColor(hmap_colored, cv2.COLOR_BGR2RGB)
    
    cam_overlay = cv2.addWeighted(orig_resized, 0.6, hmap_colored, 0.4, 0)
    cam_pil = Image.fromarray(cam_overlay)

    # Top-5 bar chart
    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor='#0f0f23')
    ax.set_facecolor('#0f0f23')
    colors = ['#7c3aed'] + ['#374151'] * 4
    bars = ax.barh(range(5)[::-1], top5_conf, color=colors, height=0.55, edgecolor='none')
    ax.set_yticks(range(5)[::-1])
    ax.set_yticklabels(top5_names, fontsize=8.5, color='#e5e7eb')
    ax.set_xlabel('Confidence (%)', fontsize=9, color='#9ca3af')
    ax.set_xlim(0, 115)
    ax.tick_params(colors='#9ca3af', labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color('#9ca3af')
    ax.tick_params(axis='x', colors='#9ca3af')
    for bar, val in zip(bars, top5_conf[::-1]):
        ax.text(val + 1.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8.5,
                fontweight='bold', color='#f9fafb')
    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, facecolor='#0f0f23')
    buf.seek(0)
    chart_pil = Image.open(buf).copy()
    plt.close()

    # Severity badge color
    severity_colors = {
        'None': ('#22c55e', '✅'),
        'Moderate': ('#f59e0b', '⚠️'),
        'High': ('#ef4444', '🔴'),
        'Critical': ('#7c3aed', '🚨'),
    }
    sev_color, sev_icon = severity_colors.get(severity, ('#6b7280', 'ℹ️'))
    status_text = "HEALTHY PLANT" if is_healthy else "DISEASE DETECTED"
    status_color = "#22c55e" if is_healthy else "#ef4444"

    steps_html = ''.join([
        f'<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:10px;">'
        f'<span style="background:#7c3aed;color:white;border-radius:50%;width:22px;height:22px;'
        f'display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:bold;flex-shrink:0;">{i+1}</span>'
        f'<span style="color:#d1d5db;font-size:14px;line-height:1.5;">{s}</span></div>'
        for i, s in enumerate(steps)
    ])

    top5_rows = ''.join([
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0;border-bottom:1px solid #1f2937;">'
        f'<span style="color:{"#a78bfa" if j==0 else "#9ca3af"};font-size:13px;">{n}</span>'
        f'<span style="color:{"#7c3aed" if j==0 else "#6b7280"};font-weight:bold;font-size:13px;">{top5_conf[j]:.1f}%</span>'
        f'</div>'
        for j, n in enumerate(top5_names)
    ])

    html = f"""
    <div style="font-family:'Segoe UI',sans-serif;background:#0f0f23;color:#f9fafb;border-radius:16px;overflow:hidden;border:1px solid #1f2937;">

      <!-- HEADER BANNER -->
      <div style="background:linear-gradient(135deg,#1e1b4b 0%,#312e81 50%,#1e1b4b 100%);
                  padding:28px 30px;border-bottom:1px solid #312e81;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
          <div>
            <div style="font-size:11px;letter-spacing:3px;color:#818cf8;text-transform:uppercase;margin-bottom:6px;">
              🌿 AI Diagnosis Report
            </div>
            <div style="font-size:26px;font-weight:800;color:#f9fafb;">{crop_name}</div>
            <div style="font-size:17px;color:#a78bfa;margin-top:4px;font-weight:500;">{display_name}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:38px;font-weight:900;color:{status_color};">{confidence:.1f}%</div>
            <div style="font-size:11px;color:#6b7280;margin-top:2px;">Confidence Score</div>
            <div style="margin-top:10px;display:inline-block;padding:5px 14px;border-radius:20px;
                        background:{status_color}22;border:1px solid {status_color};
                        color:{status_color};font-size:12px;font-weight:700;letter-spacing:1px;">
              {status_text}
            </div>
          </div>
        </div>
      </div>

      <!-- SEVERITY + INFO -->
      <div style="padding:22px 30px;border-bottom:1px solid #1f2937;">
        <div style="display:flex;gap:12px;align-items:center;margin-bottom:14px;">
          <span style="font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:2px;">Severity Level</span>
          <span style="padding:4px 14px;border-radius:20px;background:{sev_color}22;
                       border:1px solid {sev_color};color:{sev_color};
                       font-size:12px;font-weight:700;">{sev_icon} {severity}</span>
        </div>
        <div style="background:#111827;border-left:3px solid #7c3aed;border-radius:0 8px 8px 0;
                    padding:14px 18px;">
          <div style="font-size:11px;color:#6b7280;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">
            What's Happening
          </div>
          <div style="color:#d1d5db;font-size:14px;line-height:1.7;">{info}</div>
        </div>
      </div>

      <!-- TOP 5 PREDICTIONS -->
      <div style="padding:22px 30px;border-bottom:1px solid #1f2937;">
        <div style="font-size:11px;color:#6b7280;letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;">
          📊 Top 5 Predictions
        </div>
        {top5_rows}
      </div>

      <!-- TREATMENT STEPS -->
      <div style="padding:22px 30px;">
        <div style="font-size:11px;color:#6b7280;letter-spacing:3px;text-transform:uppercase;margin-bottom:16px;">
          {"🌱 Maintenance Tips" if is_healthy else "💊 Recommended Treatment"}
        </div>
        {steps_html}
      </div>

      <!-- FOOTER -->
      <div style="padding:14px 30px;background:#060612;text-align:center;
                  border-top:1px solid #1f2937;">
        <span style="font-size:11px;color:#374151;">
          🤖 Powered by MobileNetV2 + Transfer Learning &nbsp;·&nbsp; 38-Class Plant Disease Model
        </span>
      </div>
    </div>
    """
    return cam_pil, chart_pil, html

CSS = """
body, .gradio-container { background: #060612 !important; font-family: 'Segoe UI', sans-serif; }
.gr-button-primary { background: linear-gradient(135deg,#7c3aed,#4f46e5) !important;
                     border:none !important; border-radius:10px !important;
                     font-weight:700 !important; font-size:15px !important; padding:12px !important; }
.gr-button-secondary { background:#1f2937 !important; border:1px solid #374151 !important;
                        border-radius:10px !important; color:#9ca3af !important; }
.gr-image { border-radius:14px !important; border:2px solid #1f2937 !important; }
footer { display:none !important; }
"""

with gr.Blocks(css=CSS, title="🌿 Plant Disease AI") as demo:

    gr.HTML("""
    <div style="text-align:center;padding:36px 20px 20px;background:linear-gradient(180deg,#0a0a1a,#060612);">
      <div style="font-size:13px;letter-spacing:4px;color:#7c3aed;text-transform:uppercase;margin-bottom:10px;">
        AI-Powered Crop Health Analysis
      </div>
      <div style="font-size:40px;font-weight:900;color:#f9fafb;letter-spacing:-1px;">
        🌿 Plant Disease Detection
      </div>
      <div style="font-size:16px;color:#6b7280;margin-top:12px;max-width:540px;margin-left:auto;margin-right:auto;line-height:1.6;">
        Upload a leaf photo and get an instant AI diagnosis with treatment recommendations,
        Grad-CAM visual explanation, and confidence breakdown.
      </div>
      <div style="display:flex;justify-content:center;gap:20px;margin-top:20px;flex-wrap:wrap;">
        <span style="background:#111827;border:1px solid #1f2937;border-radius:20px;padding:6px 16px;
                     font-size:12px;color:#818cf8;">38 Disease Classes</span>
        <span style="background:#111827;border:1px solid #1f2937;border-radius:20px;padding:6px 16px;
                     font-size:12px;color:#818cf8;">MobileNetV2 Backbone</span>
        <span style="background:#111827;border:1px solid #1f2937;border-radius:20px;padding:6px 16px;
                     font-size:12px;color:#818cf8;">Grad-CAM Explainability</span>
        <span style="background:#111827;border:1px solid #1f2937;border-radius:20px;padding:6px 16px;
                     font-size:12px;color:#818cf8;">Expert Treatment Advice</span>
      </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div style="color:#818cf8;font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">📷 Input Image</div>')
            input_img = gr.Image(type="pil", label="Upload Leaf Image", height=320)

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear", variant="secondary", scale=1)
                submit_btn = gr.Button("🔍 Analyze", variant="primary", scale=2)

            gr.HTML("""
            <div style="margin-top:18px;background:#0d1117;border:1px solid #1f2937;border-radius:12px;padding:16px;">
              <div style="font-size:11px;color:#6b7280;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">Tips for Best Results</div>
              <div style="font-size:13px;color:#9ca3af;line-height:1.8;">
                📸 Take photo in natural daylight<br>
                🍃 Focus on a single affected leaf<br>
                🔍 Ensure the leaf fills the frame<br>
                🚫 Avoid blurry or dark images
              </div>
            </div>
            """)

        with gr.Column(scale=2):
            gr.HTML('<div style="color:#818cf8;font-size:11px;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">🔬 Grad-CAM Visual Explanation</div>')
            with gr.Row():
                cam_out = gr.Image(label="AI Focus Heatmap", height=260)
                chart_out = gr.Image(label="Confidence Chart", height=260)

            gr.HTML('<div style="color:#818cf8;font-size:11px;letter-spacing:2px;text-transform:uppercase;margin:14px 0 8px;">📋 Diagnosis Report</div>')
            report_out = gr.HTML()

    submit_btn.click(fn=analyze_leaf, inputs=[input_img], outputs=[cam_out, chart_out, report_out])
    clear_btn.click(fn=lambda: (None, None, None, ""), outputs=[input_img, cam_out, chart_out, report_out])

# This ensures Gradio uses the port assigned by Render
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))