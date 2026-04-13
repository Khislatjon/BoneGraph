"""Quick test: run LLaVA 13b on a single MURA image."""
import base64
import glob
import ollama

image_path = glob.glob(
    "data/raw/images/MURA/MURA-v1.1/train/**/*.png", recursive=True
)[0]
print("Testing on:", image_path)

with open(image_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = ollama.chat(
    model="llava:13b",
    messages=[{
        "role": "user",
        "content": (
            "You are a radiologist specialised in musculoskeletal imaging. "
            "Describe this X-ray systematically:\n"
            "1. Body part and view\n"
            "2. Cortical bone: thickness, continuity, any thinning or breaks\n"
            "3. Trabecular bone: density and pattern\n"
            "4. Any fractures, lesions, or abnormalities\n"
            "5. Overall impression in one sentence"
        ),
        "images": [img_b64],
    }],
)
print(response["message"]["content"])
