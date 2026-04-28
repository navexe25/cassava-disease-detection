# ===================== PART 2: app.py =====================
"""
Requirements:
pip install gradio plotly groq torch torchvision pillow numpy
Set environment variable before run:
GROQ_API_KEY=your_key
python app.py
"""

import gradio as gr, plotly.graph_objects as go
from PIL import Image
from groq import Groq

CLASS_NAMES = [
    'Cassava___bacterial_blight',
    'Cassava___brown_streak_disease',
    'Cassava___green_mottle',
    'Cassava___healthy',
    'Cassava___mosaic_disease'
]

client = Groq(api_key=os.getenv('GROQ_API_KEY')) if os.getenv('GROQ_API_KEY') else None
infer_model = models.resnet50(weights='IMAGENET1K_V1')
infer_model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(infer_model.fc.in_features, len(CLASS_NAMES)))
infer_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
infer_model = infer_model.to(DEVICE)
infer_model.eval()

infer_tf = val_tf

def explain(label, conf):
    if client is None:
        return 'Add GROQ_API_KEY for AI explanation.'
    prompt = f'Explain cassava disease {label} with confidence {conf:.2f}. Include symptoms, causes, treatment, prevention.'
    try:
        r = client.chat.completions.create(model='llama-3.3-70b-versatile', messages=[{'role':'user','content':prompt}], max_tokens=400)
        return r.choices[0].message.content
    except Exception as e:
        return str(e)

def predict(img):
    if img is None:
        return None, 'Upload image', None, ''
    x = infer_tf(img.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(infer_model(x), dim=1)[0].cpu().numpy()
      i = int(np.argmax(probs))
    label = CLASS_NAMES[i].replace('Cassava___','').replace('_',' ').title()
    conf = float(probs[i])
    fig = go.Figure(data=[go.Bar(x=probs, y=[c.replace('Cassava___','') for c in CLASS_NAMES], orientation='h')])
    fig.update_layout(height=400, title='Confidence Scores')
    result = f'## Prediction\n**Disease:** {label}\n\n**Confidence:** {conf*100:.2f}%'
    return img, result, fig, explain(label, conf)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown('# 🌿 Cassava Leaf Disease Detection')
    gr.Markdown('Upload a cassava leaf image for disease prediction using ResNet-50.')
    with gr.Row():
        inp = gr.Image(type='pil', label='Upload Image')
        out_img = gr.Image(label='Preview')
    btn = gr.Button('Analyze')
    result = gr.Markdown()
    chart = gr.Plot()
    exp = gr.Markdown()
    btn.click(fn=predict, inputs=inp, outputs=[out_img, result, chart, exp])

if __name__ == '__main__':
    demo.launch()
