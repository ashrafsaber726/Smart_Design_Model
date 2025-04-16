from flask import Flask, request, jsonify, render_template
from diffusers import StableDiffusionPipeline
import torch
from io import BytesIO
import base64
from PIL import Image

app = Flask(__name__)

# Load model
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
pipeline.to("cpu")  # Change to 'cuda' if you're deploying to GPU

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    try:
        image = pipeline(prompt).images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return jsonify({"image": img_str})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)