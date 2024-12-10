import mlx.core as mx
import numpy as np
import pytesseract
from ollama import chat
from PIL import Image
from tqdm import tqdm

from stable_diffusion import StableDiffusion

OLLAMA_MODEL = "dolphin-phi"
N_IMAGES = 1
CFG_WEIGHT = 7.5
STEPS = 50
SEED = 1
NEGATIVE_PROMPT = ""
FLOAT_16 = False
OUTPUT_FILENAME = "out.png"

receipt_text = pytesseract.image_to_string(Image.open("2.jpg"))
print(receipt_text)

response = chat(
    model=OLLAMA_MODEL,
    messages=[
        {
            "role": "user",
            "content": f"What the person look like who would buy this items? {receipt_text}",
        },
    ],
)
customer_description = response["message"]["content"]
print(customer_description)

sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=FLOAT_16)

sd.ensure_models_are_loaded()

# Generate the latent vectors using diffusion
latents = sd.generate_latents(
    "an apple",
    n_images=N_IMAGES,
    cfg_weight=CFG_WEIGHT,
    num_steps=STEPS,
    seed=SEED,
    negative_text=NEGATIVE_PROMPT,
)

for x_t in tqdm(latents, total=STEPS):
    mx.eval(x_t)

del sd.text_encoder
del sd.unet
del sd.sampler

decoded = []
for i in tqdm(range(0, 1, 1)):
    decoded.append(sd.decode(x_t[i : i + 1]))
    mx.eval(decoded[-1])

# Arrange them on a grid
x = mx.concatenate(decoded, axis=0)
x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
B, H, W, C = x.shape
x = x.reshape(1, B // 1, H, W, C).transpose(0, 2, 1, 3, 4)
x = x.reshape(1 * H, B // 1 * W, C)
x = (x * 255).astype(mx.uint8)

# Save them to disc
im = Image.fromarray(np.array(x))
im.save(OUTPUT_FILENAME)
