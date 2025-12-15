import os
import json
import csv
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from openai import OpenAI
import base64
import re
from time import sleep

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is not configured.")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)
# Model for image validation (use gpt-4o or gpt-4o-2024-08-06 for better precision)
VALIDATION_MODEL = "gpt-4o"  # Change to "gpt-4o-2024-08-06" or "gpt-4o" for better quality

def image_to_data_url(path: str) -> Optional[str]:
    """Converts an image to base64 data URL."""
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    
    ext = path_obj.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    else:
        mime = "image/png"
    
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except:
        return None

def validate_image_description(
    description: str,
    image_path: str,
    model: str = VALIDATION_MODEL,
    max_retries: int = 3
) -> bool:
    """
    Validates if an image matches the product description.
    Returns True if it matches, False otherwise.
    """
    data_url = image_to_data_url(image_path)
    if not data_url:
        return False
    
    content = [
        {
            "type": "text",
            "text": (
                f"Product description: {description}\n\n"
                "Does the image show this product? Answer with only 'YES' or 'NO'."
            )
        },
        {
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        }
    ]
    
    system_prompt = (
        "You are validating if an image matches a product description.\n"
        "Answer with ONLY 'YES' if the image clearly shows the described product, "
        "or 'NO' if it doesn't match or is unclear.\n"
        "Be strict: only say YES if the image clearly shows the product."
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=10
            )
            
            answer = response.choices[0].message.content.strip().upper()
            return "YES" in answer
            
        except Exception as e:
            if attempt < max_retries - 1:
                sleep(2 ** attempt)
                continue
            print(f"  ⚠️  Error validating image: {e}")
            return False
    
    return False

def load_items_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Loads items from a JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []

def combine_items(pdf_items_path: str, whatsapp_items_path: str) -> List[Dict[str, Any]]:
    """Combines items from PDF and WhatsApp."""
    all_items = []
    
    # Load PDF items
    pdf_items = load_items_from_json(pdf_items_path)
    print(f"Loaded {len(pdf_items)} items from PDF")
    
    # Load WhatsApp items
    whatsapp_items = load_items_from_json(whatsapp_items_path)
    print(f"Loaded {len(whatsapp_items)} items from WhatsApp")
    
    all_items = pdf_items + whatsapp_items
    print(f"Total items combined: {len(all_items)}")
    
    return all_items

def clean_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cleans items:
    - Removes items without price or description
    - Removes duplicates (same description + price)
    """
    cleaned = []
    seen = set()
    
    for item in items:
        # Validate that it has description and price
        description = item.get("description", "").strip()
        price_raw = item.get("price_raw", "").strip()
        price_amount = item.get("price_amount")
        
        if not description or not price_raw or price_amount is None:
            continue
        
        # Create unique key to detect duplicates
        key = (description.lower(), price_amount)
        if key in seen:
            continue
        
        seen.add(key)
        cleaned.append(item)
    
    print(f"Items after cleaning: {len(cleaned)} (removed {len(items) - len(cleaned)})")
    return cleaned

def find_image_path(image_ref: str, source: str, pdf_path: str = None, media_dir: str = None) -> Optional[str]:
    """Finds the real path of an image based on its reference."""
    if not image_ref:
        return None
    
    if source == "pdf":
        # For PDFs, search in extracted images folder
        if pdf_path:
            pdf_name = Path(pdf_path).stem
            # Search in different possible locations
            possible_dirs = [
                Path("output/pdf_images") / pdf_name,
                Path("output/pdf_images"),
            ]
            for img_dir in possible_dirs:
                if not img_dir.exists():
                    continue
                # Search exact first
                img_path = img_dir / f"{image_ref}.png"
                if img_path.exists():
                    return str(img_path)
                # Search by partial name (image_id may be in filename)
                for img_file in img_dir.glob(f"*{image_ref}*"):
                    if img_file.exists():
                        return str(img_file)
                # Also search without extension
                img_path = img_dir / image_ref
                if img_path.exists():
                    return str(img_path)
    elif source == "whatsapp":
        # For WhatsApp, search in media folder
        if media_dir:
            img_path = Path(media_dir) / image_ref
            if img_path.exists():
                return str(img_path)
    
    return None

def copy_referenced_images(items: List[Dict[str, Any]], output_images_dir: Path, media_dir: str = None):
    """Copies all referenced images to the output folder."""
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    copied = {}
    not_found = []
    
    for item in items:
        image_ref = item.get("image_ref")
        if not image_ref:
            continue
        
        # If we already copied this image, skip
        if image_ref in copied:
            continue
        
        source = item.get("source", "")
        pdf_path = item.get("pdf_path")
        
        img_path = find_image_path(image_ref, source, pdf_path, media_dir)
        
        if img_path and Path(img_path).exists():
            # Copy image
            dest_path = output_images_dir / image_ref
            try:
                shutil.copy2(img_path, dest_path)
                copied[image_ref] = str(dest_path)
            except Exception as e:
                print(f"Error copying {image_ref}: {e}")
                not_found.append(image_ref)
        else:
            not_found.append(image_ref)
    
    print(f"Images copied: {len(copied)}")
    if not_found:
        print(f"Images not found: {len(not_found)}")
    
    return copied

def validate_images(items: List[Dict[str, Any]], images_dir: Path) -> List[Dict[str, Any]]:
    """Validates images and removes references that don't match."""
    print("\n" + "=" * 60)
    print("Validating images with model")
    print("=" * 60)
    
    validated_items = []
    
    for idx, item in enumerate(items, 1):
        description = item.get("description", "")
        image_ref = item.get("image_ref")
        
        if not image_ref:
            # No image, keep the item
            validated_items.append(item)
            continue
        
        image_path = images_dir / image_ref
        if not image_path.exists():
            # Image doesn't exist, remove reference
            item_copy = item.copy()
            item_copy["image_ref"] = None
            validated_items.append(item_copy)
            print(f"[{idx}/{len(items)}] {description[:50]} - Image not found, removed")
            continue
        
        print(f"[{idx}/{len(items)}] Validating: {description[:50]}...")
        
        is_valid = validate_image_description(description, str(image_path), model=VALIDATION_MODEL)
        
        if is_valid:
            validated_items.append(item)
            print(f"  ✓ Image valid")
        else:
            item_copy = item.copy()
            item_copy["image_ref"] = None
            validated_items.append(item_copy)
            print(f"  ✗ Image doesn't match, reference removed")
        
        # Delay to avoid rate limits
        if idx < len(items):
            sleep(1)
    
    return validated_items

def clean_image_folder(images_dir: Path, referenced_images: set):
    """Removes images that are not referenced."""
    all_images = list(images_dir.glob("*"))
    removed = 0
    
    for img_path in all_images:
        if img_path.name not in referenced_images:
            try:
                img_path.unlink()
                removed += 1
            except Exception as e:
                print(f"Error removing {img_path.name}: {e}")
    
    print(f"Images removed (not referenced): {removed}")

def create_csv(items: List[Dict[str, Any]], csv_path: Path):
    """Creates the final CSV with required columns."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["description", "image_ref", "price"])
        writer.writeheader()
        
        for item in items:
            writer.writerow({
                "description": item.get("description", ""),
                "image_ref": item.get("image_ref", ""),
                "price": item.get("price_amount", "")
            })
    
    print(f"CSV created: {csv_path}")

def calculate_stats(csv_path: Path):
    """Calculates statistics of the final dataset."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "=" * 60)
    print("FINAL DATASET STATISTICS")
    print("=" * 60)
    print(f"Total observations: {len(df)}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    print(f"\nPrice:")
    print(f"  Mean: ${df['price'].mean():.2f}")
    print(f"  Median: ${df['price'].median():.2f}")
    print(f"  Minimum: ${df['price'].min():.2f}")
    print(f"  Maximum: ${df['price'].max():.2f}")
    print(f"  Standard deviation: ${df['price'].std():.2f}")
    print(f"\nItems with image: {df['image_ref'].notna().sum()}")
    print(f"Items without image: {df['image_ref'].isna().sum()}")

def main():
    # Paths
    pdf_items_path = "output/pdf_items.json"
    whatsapp_items_path = "output/whatsapp_items.json"
    dataset_dir = Path("dataset_final")
    images_dir = dataset_dir / "images"
    csv_path = dataset_dir / "dataset.csv"
    media_dir = "WhatsApp Chat - Sloan Buy _ Sell 26s + 25s (1)"
    
    print("=" * 60)
    print("BUILDING FINAL DATASET")
    print("=" * 60)
    
    # 1. Combine items
    print("\n[1/5] Combining items from PDF and WhatsApp...")
    all_items = combine_items(pdf_items_path, whatsapp_items_path)
    
    # 2. Clean items
    print("\n[2/5] Cleaning items...")
    cleaned_items = clean_items(all_items)
    
    # 3. Copy referenced images
    print("\n[3/5] Copying referenced images...")
    copied_images = copy_referenced_images(cleaned_items, images_dir, media_dir)
    
    # 4. Validate images with model
    print(f"\n[4/5] Validating images with model {VALIDATION_MODEL}...")
    validated_items = validate_images(cleaned_items, images_dir)
    
    # 5. Clean image folder (remove non-referenced)
    print("\n[5/5] Cleaning image folder...")
    referenced_set = {item.get("image_ref") for item in validated_items if item.get("image_ref")}
    clean_image_folder(images_dir, referenced_set)
    
    # 6. Create final CSV
    print("\n[6/6] Creating final CSV...")
    create_csv(validated_items, csv_path)
    
    # 7. Statistics
    calculate_stats(csv_path)
    
    print("\n" + "=" * 60)
    print("PROCESS COMPLETED")
    print("=" * 60)
    print(f"Dataset saved to: {dataset_dir}")
    print(f"CSV: {csv_path}")
    print(f"Images: {images_dir}")

if __name__ == "__main__":
    main()

