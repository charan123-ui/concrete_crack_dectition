"""
Dataset Generator & Utility for Concrete Crack Detection
=========================================================
Generates balanced train, validation, and test splits of photographic concrete surface images:
  - Class Positive: Cracked Concrete
  - Class Negative: Sound / Non-Cracked Concrete
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_concrete_texture(width=224, height=224, base_gray=180, roughness=22):
    """Generates realistic concrete surface texture with aggregates and pores."""
    base = np.random.normal(loc=base_gray, scale=roughness, size=(height, width)).clip(60, 235)
    
    # Low frequency surface variation (lighting / formwork marks)
    low_res = np.random.normal(0, 12, size=(height // 16, width // 16))
    low_img = Image.fromarray(low_res.astype(np.float32), mode='F').resize((width, height), resample=Image.BICUBIC)
    low_noise = np.array(low_img)
    
    concrete = np.clip(base + low_noise, 50, 240).astype(np.uint8)
    img = Image.fromarray(concrete, mode='L').convert('RGB')
    
    # Aggregate speckles / pores
    draw = ImageDraw.Draw(img)
    num_specks = random.randint(30, 80)
    for _ in range(num_specks):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        radius = random.choice([1, 1, 2])
        shade = random.randint(50, 120)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(shade, shade, shade))
        
    return img

def add_crack_to_surface(img):
    """Renders a prominent, realistic fracture crack line across concrete with branches."""
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    w, h = img.size
    
    axis = random.choice(['vertical', 'horizontal', 'diagonal'])
    if axis == 'vertical':
        x = random.randint(w // 4, 3 * w // 4)
        y = 0
    elif axis == 'horizontal':
        x = 0
        y = random.randint(h // 4, 3 * h // 4)
    else:
        x = random.randint(0, w // 3)
        y = 0

    points = [(x, y)]
    current_x, current_y = x, y
    steps = random.randint(35, 55)
    base_width = random.uniform(2.2, 4.0)
    
    for _ in range(steps):
        if axis == 'vertical':
            current_y += h / steps
            current_x += random.uniform(-5, 5)
        elif axis == 'horizontal':
            current_x += w / steps
            current_y += random.uniform(-5, 5)
        else:
            current_x += w / steps + random.uniform(-4, 4)
            current_y += h / steps + random.uniform(-4, 4)
            
        current_x = max(2, min(w - 3, current_x))
        current_y = max(2, min(h - 3, current_y))
        points.append((current_x, current_y))

    # Dark crack fissure
    crack_color = (random.randint(10, 30), random.randint(10, 30), random.randint(10, 30))
    
    # Draw primary fracture
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]
        draw.line([pt1, pt2], fill=crack_color, width=int(base_width + 1))
        
        # Branch crack
        if random.random() < 0.28 and 4 < i < len(points) - 4:
            bx, by = pt1
            branch_len = random.randint(18, 40)
            b_angle = random.uniform(-1.2, 1.2)
            bx2 = bx + branch_len * np.cos(b_angle)
            by2 = by + branch_len * np.sin(b_angle)
            draw.line([(bx, by), (bx2, by2)], fill=crack_color, width=max(1, int(base_width - 1)))
            
    blurred = img_draw.filter(ImageFilter.GaussianBlur(radius=0.35))
    return blurred

def build_dataset(base_dir=None, train_per_class=200, val_per_class=50, test_per_class=40):
    """Builds balanced train, val, and test splits with Positive and Negative classes."""
    if base_dir is None:
        base_dir = os.path.join(SCRIPT_DIR, "dataset")
        
    random.seed(42)
    np.random.seed(42)

    splits = {
        'train': train_per_class,
        'val': val_per_class,
        'test': test_per_class
    }
    
    for split, count in splits.items():
        pos_dir = os.path.join(base_dir, split, "Positive")
        neg_dir = os.path.join(base_dir, split, "Negative")
        os.makedirs(pos_dir, exist_ok=True)
        os.makedirs(neg_dir, exist_ok=True)
        
        print(f"Creating {split} dataset ({count} Positive, {count} Negative)...")
        # Negative (Sound Concrete)
        for i in range(count):
            img = create_concrete_texture()
            img.save(os.path.join(neg_dir, f"sound_{i+1:04d}.jpg"), quality=95)
            
        # Positive (Cracked Concrete)
        for i in range(count):
            img = create_concrete_texture()
            cracked_img = add_crack_to_surface(img)
            cracked_img.save(os.path.join(pos_dir, f"crack_{i+1:04d}.jpg"), quality=95)

    # Standalone demo test samples
    demo_dir = os.path.join(SCRIPT_DIR, "test_samples")
    os.makedirs(demo_dir, exist_ok=True)
    for idx in range(1, 4):
        img = create_concrete_texture(base_gray=175)
        cracked = add_crack_to_surface(img)
        cracked.save(os.path.join(demo_dir, f"crack_sample_{idx}.jpg"), quality=95)
        
    for idx in range(1, 4):
        img = create_concrete_texture(base_gray=185)
        img.save(os.path.join(demo_dir, f"sound_sample_{idx}.jpg"), quality=95)
        
    print(f"[+] Dataset successfully created in '{base_dir}' and sample images in '{demo_dir}'.")

if __name__ == "__main__":
    build_dataset()
