#!/usr/bin/env python3
"""
Batch inference script for testing the instruct-pix2pix model on test images.
Generates an HTML comparison page for easy viewing of results.
"""

import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionInstructPix2PixPipeline


def get_test_images(test_dir: str) -> list[Path]:
    """Get all input images (those not ending with _output.png)."""
    test_path = Path(test_dir)
    all_images = sorted(test_path.glob("*.png"))
    # Filter out the ground truth outputs
    input_images = [img for img in all_images if not img.stem.endswith("_output")]
    return input_images


def generate_html_report(
    test_dir: str,
    output_dir: str,
    image_names: list[str],
    times: list[float],
) -> str:
    """Generate an HTML comparison page."""
    total_time = sum(times)
    avg_time = total_time / len(times) if times else 0
    total = len(image_names)

    html_parts = [
        f"""<!DOCTYPE html>
<html>
<head>
    <title>Instruct Pix2Pix Test Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            color: #fff;
            margin-bottom: 30px;
        }}
        .stats {{
            text-align: center;
            margin-bottom: 30px;
            color: #aaa;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(900px, 1fr));
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }}
        .comparison {{
            background: #16213e;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .comparison h3 {{
            margin: 0 0 10px 0;
            color: #e94560;
            font-size: 14px;
        }}
        .images {{
            display: flex;
            gap: 10px;
        }}
        .image-container {{
            flex: 1;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            height: 200px;
            object-fit: contain;
            border-radius: 8px;
            background: #0f0f23;
        }}
        .image-container p {{
            margin: 8px 0 0 0;
            font-size: 12px;
            color: #888;
        }}
        .time {{
            font-size: 11px;
            color: #4ade80;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <h1>Instruct Pix2Pix Test Results</h1>
    <div class="stats">
        Total images: {total} | Average time: {avg_time:.2f}s | Total time: {total_time:.1f}s
    </div>
    <div class="grid">
"""
    ]

    for name, t in zip(image_names, times):
        input_path = f"{test_dir}/{name}.png"
        gt_path = f"{test_dir}/{name}_output.png"
        generated_path = f"{output_dir}/{name}_generated.png"

        html_parts.append(
            f"""
        <div class="comparison">
            <h3>{name}</h3>
            <div class="images">
                <div class="image-container">
                    <img src="{input_path}" alt="Input">
                    <p>Input</p>
                </div>
                <div class="image-container">
                    <img src="{generated_path}" alt="Generated">
                    <p>Generated</p>
                    <p class="time">{t:.2f}s</p>
                </div>
                <div class="image-container">
                    <img src="{gt_path}" alt="Ground Truth">
                    <p>Ground Truth</p>
                </div>
            </div>
        </div>
"""
        )

    html_parts.append(
        """
    </div>
</body>
</html>
"""
    )
    return "".join(html_parts)


def main():
    # Configuration
    test_dir = "/data/robotsmith/task03_flatten/wm_dataset/test"
    model_path = (
        "/home/amli/research/diffusers/examples/instruct_pix2pix/robotsmith-flatten-wm"
    )
    output_dir = "/home/amli/research/diffusers/examples/instruct_pix2pix/test_results"
    prompt = "flatten the dough to a height smaller than 0.03"

    # Inference parameters
    num_inference_steps = 20
    image_guidance_scale = 1.5
    guidance_scale = 10
    seed = 0
    batch_size = 4  # Process multiple images at once

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}...")
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to("cuda")

    # Get test images
    input_images = get_test_images(test_dir)
    print(f"Found {len(input_images)} test images")
    print(f"Processing in batches of {batch_size}")

    # Process images in batches
    image_names = []
    times = []

    num_batches = (len(input_images) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(input_images))
        batch_paths = input_images[batch_start:batch_end]

        # Load batch of images
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        batch_prompts = [prompt] * len(batch_images)

        # Reset generator for reproducibility
        generator = torch.Generator("cuda").manual_seed(seed)

        # Generate batch
        start_time = time.time()

        edited_images = pipeline(
            batch_prompts,
            image=batch_images,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images

        elapsed = time.time() - start_time
        time_per_image = elapsed / len(batch_images)

        # Save results
        for img_path, edited_image in zip(batch_paths, edited_images):
            output_path = Path(output_dir) / f"{img_path.stem}_generated.png"
            edited_image.save(output_path)
            image_names.append(img_path.stem)
            times.append(time_per_image)

    # Generate HTML report
    print("\nGenerating HTML report...")
    html_content = generate_html_report(test_dir, output_dir, image_names, times)
    html_path = Path(output_dir) / "comparison.html"
    html_path.write_text(html_content)

    print(f"\nDone! Results saved to: {output_dir}")
    print(f"View results: open {html_path}")
    print(f"\nSummary:")
    print(f"  - Images processed: {len(image_names)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Average time per image: {sum(times)/len(times):.2f}s")
    print(f"  - Total time: {sum(times):.1f}s")


if __name__ == "__main__":
    main()
