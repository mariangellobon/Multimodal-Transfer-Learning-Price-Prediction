import os
import re
import json
import base64
import uuid
import time
from time import sleep
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import fitz  # PyMuPDF
    import pdfplumber
    from openai import OpenAI
except ImportError:
    print("ERROR: Missing dependencies. Install with:")
    print("  pip install PyMuPDF pdfplumber openai")
    exit(1)

# ======================================
# CONFIG
# ======================================

OPENAI_MODEL = "gpt-4o-mini"  # Default model (more economical)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY is not configured. Set it with:")
    print("  export OPENAI_API_KEY='your-api-key'")
    print("  Or on Windows: set OPENAI_API_KEY=your-api-key")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# ======================================
# COMMON UTILITIES (IMAGES + GPT)
# ======================================

def image_to_data_url(path: str) -> str:
    """
    Reads a local image and converts it to base64 data URL (for image_url).
    """
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
    
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

# ======================================
# PASS 1: TEXT ONLY - Sales extraction
# ======================================

# Schema for first pass of WhatsApp (text only)
TEXT_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "message_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "IDs of messages where this product is being sold."
                    },
                    "sender": {
                        "type": "string",
                        "description": "Name of the person selling this item."
                    },
                    "description": {
                        "type": "string",
                        "description": "Clean description of the product being sold (WITHOUT price information)."
                    },
                    "price_raw": {
                        "type": "string",
                        "description": "Price string exactly as shown in the messages."
                    },
                    "price_amount": {
                        "type": ["number", "null"],
                        "description": "Numeric value if parsable, else null."
                    },
                    "currency": {
                        "type": "string",
                        "description": "Currency code or symbol, e.g. USD, ARS, $, unknown."
                    },
                    "candidate_image_filenames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of ALL image filenames (from <attached: filename>) that have ANY possibility of being related to this product. Be GENEROUS and INCLUSIVE - include all images from the SAME sender that appear in the context (before, during, or after the sale message). A second pass will later examine the actual images and decide which one matches. Only use empty array if there are absolutely no images from the same sender in the context."
                    }
                },
                "required": ["message_ids", "sender", "description", "price_raw", "price_amount", "currency", "candidate_image_filenames"],
                "additionalProperties": False
            }
        }
    },
    "required": ["items"],
    "additionalProperties": False
}

def call_gpt_text_extraction(
    text_block: str,
    model: str = OPENAI_MODEL,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    First pass: Text only. Identifies sales and extracts information.
    Does NOT look at images, only uses chat context.
    """
    
    system_prompt = (
        "You analyze WhatsApp chat messages to identify items being sold.\n"
        "You MUST return a JSON object strictly following the given JSON schema.\n\n"
        "CRITICAL RULES:\n"
        "- ONLY extract items that are clearly being offered FOR SALE.\n"
        "- IGNORE messages asking to buy (WTB / looking for / 'anyone selling').\n"
        "- EVERY item MUST have a price - if there's no clear price, DO NOT extract it.\n"
        "- 'description' must describe WHAT is being sold (e.g. 'IKEA ÄSPERÖD coffee table'), "
        "  NOT generic phrases like 'Selling', 'Selling for', 'Items for sale', 'OBO', 'Price'.\n"
        "- 'description' must NOT include the price - remove all price information from description.\n"
        "- 'price_raw' should be the exact price string (e.g. '$25, OBO', '30$', '$80 each').\n"
        "- 'price_amount' is the numeric value if you can parse it (e.g. '$25, OBO' -> 25.0), else null.\n"
        "- 'currency' is 'USD', 'ARS', '$', or 'unknown'.\n"
        "- 'candidate_image_filenames': IMPORTANT - Be GENEROUS and INCLUSIVE when selecting candidate images.\n"
        "  Include ALL images that have ANY possibility of being related to the product, even if you're not sure.\n"
        "  A second pass will later examine the actual images and decide which one (if any) truly matches.\n"
        "  Your job is to provide a comprehensive list of candidates, not to be the final judge.\n"
        "- Include images sent by the SAME sender who is selling the product.\n"
        "- Consider timing: images sent before, during, or after the sale message (within reasonable time windows).\n"
        "- Consider context: if the seller mentions multiple items or shows a collection, include all related images.\n"
        "- If images are sent in a sequence around the sale message, include all of them.\n"
        "- You can see image filenames in the text as '<IMAGE ATTACHMENT: filename>'.\n"
        "- When in doubt, include the image - let the second pass decide.\n"
        "- Only use an empty array if there are absolutely NO images from the same sender in the context.\n"
        "- NEVER invent products or prices that are not present in the text.\n"
        "- If a message mentions selling but has no price, DO NOT extract it.\n"
    )
    
    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": text_block
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "text_extraction",
                        "strict": True,
                        "schema": TEXT_EXTRACTION_SCHEMA
                    }
                }
            )
            
            raw_text = response.choices[0].message.content
            return json.loads(raw_text)
            
        except Exception as e:
            error_str = str(e)
            
            # Rate limit error
            if "rate_limit" in error_str.lower() or "429" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 10 * (2 ** attempt)
                    wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                    if wait_match:
                        wait_time = float(wait_match.group(1)) + 2
                    
                    print(f"  ⚠️  Rate limit reached. Waiting {wait_time:.1f}s before retrying...")
                    sleep(wait_time)
                    continue
                else:
                    print(f"  ✗ Error after {max_retries} attempts: Rate limit")
                    return {"items": []}
            else:
                print(f"  ✗ Error calling GPT: {error_str}")
                if attempt < max_retries - 1:
                    sleep(2 ** attempt)
                    continue
                return {"items": []}
    
    return {"items": []}

# ======================================
# PASS 2: MULTIMODAL - Image validation
# ======================================

# Schema for second pass (multimodal)
IMAGE_VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "image_ref": {
            "type": ["string", "null"],
            "description": "Filename of the ONE image that best matches the product, or null if none match."
        }
    },
    "required": ["image_ref"],
    "additionalProperties": False
}

def call_gpt_image_validation(
    description: str,
    price_raw: str,
    candidate_image_paths: Dict[str, str],  # {image_id_or_filename: path}
    model: str = OPENAI_MODEL,
    max_retries: int = 5
) -> Optional[str]:
    """
    Second pass: Multimodal. Validates if any candidate image actually shows the product.
    Returns the image_id/filename of the best matching image, or None.
    """
    
    if not candidate_image_paths:
        return None
    
    content = [
        {
            "type": "text",
            "text": (
                f"Product description: {description}\n"
                f"Price: {price_raw}\n\n"
                "Candidate images:\n" +
                "\n".join([f"- {img_id}" for img_id in candidate_image_paths.keys()]) +
                "\n\nDetermine which ONE image (if any) actually shows this product being sold. "
                "Return the image identifier (same as shown above) if it matches, or null if none of the images clearly match the product."
            )
        }
    ]
    
    # Add all candidate images
    for img_id, path in candidate_image_paths.items():
        data_url = image_to_data_url(path)
        if data_url:
            content.append({
                "type": "text",
                "text": f"[IMAGE: {img_id}]"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            })
    
    system_prompt = (
        "You validate if candidate images match a product being sold.\n"
        "You MUST return a JSON object with 'image_ref' set to:\n"
        "- The image identifier (same as shown in the candidate list) of the ONE image that best matches the product, OR\n"
        "- null if NONE of the candidate images clearly show the product\n\n"
        "Rules:\n"
        "- Be strict: only assign an image if it clearly shows the product described.\n"
        "- If multiple images show the product, choose the one that best represents it.\n"
        "- If no image matches, return null - this is perfectly acceptable.\n"
        "- NEVER assign a random image just to fill the field.\n"
        "- Return the EXACT image identifier as shown in the candidate list (filename or image_id).\n"
    )
    
    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "image_validation",
                        "strict": True,
                        "schema": IMAGE_VALIDATION_SCHEMA
                    }
                }
            )
            
            raw_text = response.choices[0].message.content
            result = json.loads(raw_text)
            return result.get("image_ref")
            
        except Exception as e:
            error_str = str(e)
            
            # Rate limit error
            if "rate_limit" in error_str.lower() or "429" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 10 * (2 ** attempt)
                    wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                    if wait_match:
                        wait_time = float(wait_match.group(1)) + 2
                    
                    print(f"  ⚠️  Rate limit alcanzado. Esperando {wait_time:.1f}s antes de reintentar...")
                    sleep(wait_time)
                    continue
                else:
                    print(f"  ✗ Error después de {max_retries} intentos: Rate limit")
                    return None
            else:
                print(f"  ✗ Error llamando a GPT: {error_str}")
                if attempt < max_retries - 1:
                    sleep(2 ** attempt)
                    continue
                return None
    
    return None

# ======================================
# PART 1: WHATSAPP + IMAGES
# ======================================

WHATSAPP_LINE_RE = re.compile(
    r'^\[(?P<date>[^,]+), (?P<time>[^\]]+)\] (?P<sender>[^:]+): ?(?P<message>.*)$'
)
ATTACHED_RE = re.compile(r'<attached:\s*([^>]+)>', re.IGNORECASE)

@dataclass
class WhatsAppMsg:
    id: int
    timestamp: str
    sender: str
    text: str
    attachment: Optional[str] = None

def parse_whatsapp_chat(path: str) -> List[WhatsAppMsg]:
    messages: List[WhatsAppMsg] = []
    current: Optional[WhatsAppMsg] = None
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            line_stripped = line.lstrip("\u200e").lstrip("\ufeff")
            m = WHATSAPP_LINE_RE.match(line_stripped)
            if m:
                if current is not None:
                    messages.append(current)
                
                date = m.group("date").strip()
                time = m.group("time").strip()
                sender = m.group("sender").strip()
                msg_text = m.group("message").strip()
                ts = f"{date} {time}"
                
                att_match = ATTACHED_RE.search(msg_text)
                attachment = att_match.group(1).strip() if att_match else None
                
                current = WhatsAppMsg(
                    id=len(messages),
                    timestamp=ts,
                    sender=sender,
                    text=msg_text,
                    attachment=attachment
                )
            else:
                if current is not None:
                    current.text += "\n" + line_stripped
    
    if current is not None:
        messages.append(current)
    
    return messages

def build_whatsapp_block_text(msgs: List[WhatsAppMsg], all_msgs: List[WhatsAppMsg] = None, context_window: int = 10) -> str:
    """
    Builds a readable text block for GPT.
    Includes information about attached images.
    If all_msgs is provided, includes additional context (messages before/after) to help identify images.
    """
    lines = []
    
    # If we have all messages, include additional context
    if all_msgs is not None:
        block_ids = [m.id for m in msgs]
        if block_ids:
            min_id = min(block_ids)
            max_id = max(block_ids)
            start_id = max(0, min_id - context_window)
            end_id = min(len(all_msgs) - 1, max_id + context_window)
            
            # Include context messages
            for i in range(start_id, end_id + 1):
                m = all_msgs[i]
                is_in_block = min_id <= m.id <= max_id
                prefix = "[CONTEXT] " if not is_in_block else ""
                
                if m.attachment:
                    text = f"<IMAGE ATTACHMENT: {m.attachment}>"
                else:
                    text = m.text
                lines.append(f"{prefix}[{m.id}] {m.timestamp} - {m.sender}: {text}")
    else:
        # No additional context, just the block
        for m in msgs:
            if m.attachment:
                text = f"<IMAGE ATTACHMENT: {m.attachment}>"
            else:
                text = m.text
            lines.append(f"[{m.id}] {m.timestamp} - {m.sender}: {text}")
    
    return "\n".join(lines)

def chunk_messages(messages: List[WhatsAppMsg],
                   chunk_size: int = 25,
                   overlap: int = 5) -> List[List[WhatsAppMsg]]:
    chunks = []
    n = len(messages)
    i = 0
    while i < n:
        chunk = messages[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def extract_from_whatsapp(
    chat_path: str,
    media_dir: str,
    output_json: str,
    model: str = OPENAI_MODEL
):
    print("=" * 60)
    print("PASS 1: Text extraction (identify sales)")
    print("=" * 60)
    
    msgs = parse_whatsapp_chat(chat_path)
    chunks = chunk_messages(msgs, chunk_size=25, overlap=5)
    
    # Initialize empty JSON file
    all_items = []
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If file already exists, load it to continue
    if output_path.exists():
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                # If it already has image_ref, it means it passed through both passes
                if existing_data and "image_ref" in existing_data[0]:
                    print("[INFO] File already fully processed. Delete the file to reprocess.")
                    return
                all_items = existing_data
            print(f"[INFO] Continuing from {len(all_items)} items already extracted")
        except:
            all_items = []
    
    # PASS 1: Text only
    for idx, chunk in enumerate(chunks):
        print(f"\n[PASS 1] Processing chunk {idx+1}/{len(chunks)}...")
        # Include additional context to help identify candidate images
        text_block = build_whatsapp_block_text(chunk, all_msgs=msgs, context_window=15)
        
        result = call_gpt_text_extraction(
            text_block=text_block,
            model=model
        )
        
        chunk_items = []
        for item in result.get("items", []):
            # Validate that it has a price
            if not item.get("price_raw") or item.get("price_raw", "").lower() in ["unknown", "none", ""]:
                print(f"  ⚠️  Item without valid price, skipping: {item.get('description', 'N/A')[:50]}")
                continue
            
            item["source"] = "whatsapp"
            item["chat_path"] = chat_path
            chunk_items.append(item)
            all_items.append(item)
        
        # Save incrementally after each chunk
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Items extracted in this chunk: {len(chunk_items)}")
        print(f"  ✓ Total accumulated: {len(all_items)} items")
        
        # Delay between chunks to avoid rate limits
        if idx < len(chunks) - 1:
            sleep(3)
    
    print(f"\n[PASS 1 COMPLETED] Total items extracted: {len(all_items)}")
    
    # PASS 2: Image validation
    print("\n" + "=" * 60)
    print("PASS 2: Image validation (multimodal)")
    print("=" * 60)
    
    for idx, item in enumerate(all_items):
        print(f"\n[PASS 2] Processing item {idx+1}/{len(all_items)}: {item.get('description', 'N/A')[:50]}")
        
        candidate_filenames = item.get("candidate_image_filenames", [])
        
        if not candidate_filenames:
            item["image_ref"] = None
            print(f"  → No candidate images, image_ref = null")
        else:
            # Build candidate image paths
            candidate_paths = {}
            for fname in candidate_filenames:
                # Only JPG/JPEG
                if fname.lower().endswith(('.jpg', '.jpeg')):
                    path = Path(media_dir) / fname
                    if path.exists():
                        candidate_paths[fname] = str(path)
            
            if not candidate_paths:
                item["image_ref"] = None
                print(f"  → Candidate images not found, image_ref = null")
            else:
                print(f"  → Validating {len(candidate_paths)} candidate images...")
                
                image_ref = call_gpt_image_validation(
                    description=item["description"],
                    price_raw=item["price_raw"],
                    candidate_image_paths=candidate_paths,
                    model=model
                )
                
                item["image_ref"] = image_ref
                if image_ref:
                    print(f"  ✓ Image assigned: {image_ref}")
                else:
                    print(f"  → No image matches, image_ref = null")
        
        # Save incrementally after each item
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        
        # Delay between items
        if idx < len(all_items) - 1:
            sleep(2)
    
    # Clean temporary field
    for item in all_items:
        if "candidate_image_filenames" in item:
            del item["candidate_image_filenames"]
    
    # Final save
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Process completed. Saved to {output_json}")
    print(f"[OK] Total final items: {len(all_items)}")

# ======================================
# PART 2: PDFs → INDIVIDUAL IMAGES + JSON
# ======================================

@dataclass
class PdfImageRecord:
    image_id: str
    page: int
    path: str
    bbox: tuple  # (x0, y0, x1, y1) bounding box coordinates

def extract_images_from_pdf(pdf_path: str,
                            output_folder: str) -> List[PdfImageRecord]:
    """
    Extracts ALL embedded images from the PDF (not the full page).
    Returns a list with (image_id, page, path, bbox).
    Includes spatial coordinates (bounding box) for each image.
    """
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    
    image_records: List[PdfImageRecord] = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        
        for i, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            
            image_id = f"{os.path.basename(pdf_path)}_p{page_num}_img{i}_{uuid.uuid4().hex[:6]}"
            image_filename = f"{image_id}.png"
            output_path = os.path.join(output_folder, image_filename)
            
            if pix.n < 5:
                pix.save(output_path)
            else:
                pix_converted = fitz.Pixmap(fitz.csRGB, pix)
                pix_converted.save(output_path)
                pix_converted = None
            
            pix = None
            
            # Get image bounding box
            bbox = (0, 0, 0, 0)  # Default
            try:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
            except:
                # If cannot be obtained, use default coordinates
                pass
            
            image_records.append(PdfImageRecord(
                image_id=image_id,
                page=page_num,
                path=output_path,
                bbox=bbox
            ))
    
    doc.close()
    return image_records

def extract_text_by_page(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from each page with spatial information (coordinates).
    Includes words with their positions to help associate products with prices.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Full text
            full_text = page.extract_text() or ""
            
            # Words with coordinates
            words = page.extract_words()
            
            # Organize words by position (for spatial context)
            words_with_pos = []
            for word in words:
                # Extract only existing keys
                word_data = {
                    "text": word.get("text", ""),
                    "x0": word.get("x0", 0),
                    "x1": word.get("x1", 0),
                }
                # Add optional keys if they exist
                if "y0" in word:
                    word_data["y0"] = word["y0"]
                if "y1" in word:
                    word_data["y1"] = word["y1"]
                if "top" in word:
                    word_data["top"] = word["top"]
                if "bottom" in word:
                    word_data["bottom"] = word["bottom"]
                
                words_with_pos.append(word_data)
            
            pages_text.append({
                "page": i,
                "text": full_text,
                "words": words_with_pos,
                "page_width": page.width,
                "page_height": page.height
            })
    return pages_text

# Schema for first pass of PDFs (text only)
PDF_TEXT_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Clean description of the product being sold (WITHOUT price information)."
                    },
                    "price_raw": {
                        "type": "string",
                        "description": "Price string exactly as shown in the page."
                    },
                    "price_amount": {
                        "type": ["number", "null"],
                        "description": "Numeric value if parsable, else null."
                    },
                    "currency": {
                        "type": "string",
                        "description": "Currency code or symbol, e.g. USD, ARS, $, unknown."
                    }
                },
                "required": ["description", "price_raw", "price_amount", "currency"],
                "additionalProperties": False
            }
        }
    },
    "required": ["items"],
    "additionalProperties": False
}

def call_gpt_pdf_text_extraction(
    page_data: Dict[str, Any],
    page_num: int,
    model: str = OPENAI_MODEL,
    max_retries: int = 5
) -> Dict[str, Any]:
    """
    First pass for PDFs: Text with spatial information. Extracts products and prices from a page.
    Uses coordinates to associate products with their prices.
    """
    
    page_text = page_data["text"]
    words = page_data.get("words", [])
    page_width = page_data.get("page_width", 0)
    page_height = page_data.get("page_height", 0)
    
    # Build spatial representation of text
    # Group nearby words into "blocks" to help the model
    spatial_info = ""
    if words:
        spatial_info = "\n\nSpatial layout (words with coordinates, Y increases downward):\n"
        spatial_info += f"Page dimensions: {page_width:.1f} x {page_height:.1f} points\n"
        spatial_info += "Words grouped by vertical position (top to bottom):\n\n"
        
        # Group words by vertical position (approximately on the same "line")
        words_by_line = {}
        for word in words:
            # Use "top" if exists, else "y0", else use 0
            y_pos = word.get("top", word.get("y0", 0))
            # Round Y to groups of ~10 points to group lines
            line_key = int(y_pos / 10) * 10
            if line_key not in words_by_line:
                words_by_line[line_key] = []
            words_by_line[line_key].append(word)
        
        # Sort by vertical position
        for line_y in sorted(words_by_line.keys()):
            line_words = words_by_line[line_y]
            line_text = " ".join([w.get("text", "") for w in sorted(line_words, key=lambda x: x.get("x0", 0))])
            # Calculate average Y using available key
            y_values = [w.get("top", w.get("y0", 0)) for w in line_words]
            avg_y = sum(y_values) / len(y_values) if y_values else 0
            x_values = [w.get("x0", 0) for w in line_words]
            avg_x = sum(x_values) / len(x_values) if x_values else 0
            spatial_info += f"  Y={avg_y:.1f}, X={avg_x:.1f}: {line_text}\n"
    
    system_prompt = (
        "You analyze a PDF page from a moving-sale or items-for-sale flyer to identify products being sold.\n"
        "You MUST return a JSON object strictly following the given JSON schema.\n\n"
        "CRITICAL RULES:\n"
        "- Extract ALL items that are clearly being offered FOR SALE on this page.\n"
        "- EVERY item MUST have a price - if there's no clear price, DO NOT extract it.\n"
        "- Use the SPATIAL LAYOUT information to associate products with their prices.\n"
        "- Products and prices that are close together vertically (similar Y coordinates) are likely related.\n"
        "- If multiple products appear on the page, each should have its own entry with its corresponding price.\n"
        "- 'description' must describe WHAT is being sold (e.g. 'IKEA ÄSPERÖD coffee table'), "
        "  NOT generic phrases like 'Selling', 'Selling for', 'Items for sale', 'OBO', 'Price'.\n"
        "- 'description' must NOT include the price - remove all price information from description.\n"
        "- 'price_raw' should be the exact price string (e.g. '$25, OBO', '30$', '$80 each').\n"
        "- 'price_amount' is the numeric value if you can parse it (e.g. '$25, OBO' -> 25.0), else null.\n"
        "- 'currency' is 'USD', 'ARS', '$', or 'unknown'.\n"
        "- If a product is mentioned but has no price, DO NOT extract it.\n"
        "- Extract multiple products if the page shows multiple items for sale.\n"
        "- Pay attention to vertical proximity: items and prices on the same horizontal level (similar Y) are likely paired.\n"
    )
    
    user_content = (
        f"PDF Page {page_num}:\n\n"
        f"Full text:\n{page_text}\n"
        f"{spatial_info}\n\n"
        "Extract all products for sale from this page, using spatial layout to match products with their prices."
    )
    
    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "pdf_text_extraction",
                        "strict": True,
                        "schema": PDF_TEXT_EXTRACTION_SCHEMA
                    }
                }
            )
            
            raw_text = response.choices[0].message.content
            return json.loads(raw_text)
            
        except Exception as e:
            error_str = str(e)
            
            # Rate limit error
            if "rate_limit" in error_str.lower() or "429" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 10 * (2 ** attempt)
                    wait_match = re.search(r'try again in ([\d.]+)s', error_str)
                    if wait_match:
                        wait_time = float(wait_match.group(1)) + 2
                    
                    print(f"  ⚠️  Rate limit reached. Waiting {wait_time:.1f}s before retrying...")
                    sleep(wait_time)
                    continue
                else:
                    print(f"  ✗ Error after {max_retries} attempts: Rate limit")
                    return {"items": []}
            else:
                print(f"  ✗ Error calling GPT: {error_str}")
                if attempt < max_retries - 1:
                    sleep(2 ** attempt)
                    continue
                return {"items": []}
    
    return {"items": []}

def extract_from_pdf(
    pdf_path: str,
    images_dir: str,
    output_json: str,
    model: str = OPENAI_MODEL
):
    print("=" * 60)
    print("PREPROCESSING: PDF image extraction")
    print("=" * 60)
    
    # PREPROCESSING: Extract individual images
    print(f"Extracting individual images from {pdf_path}...")
    try:
        image_records = extract_images_from_pdf(pdf_path, images_dir)
        print(f"Images extracted: {len(image_records)}")
    except Exception as e:
        print(f"Error extracting images: {e}")
        image_records = []
    
    # Group images by page
    images_by_page: Dict[int, List[PdfImageRecord]] = {}
    for img_record in image_records:
        if img_record.page not in images_by_page:
            images_by_page[img_record.page] = []
        images_by_page[img_record.page].append(img_record)
    
    print(f"Pages with images: {len(images_by_page)}")
    
    # Extract text by page
    print("\nExtracting text by page...")
    try:
        pages_text = extract_text_by_page(pdf_path)
        print(f"Total pages: {len(pages_text)}")
    except Exception as e:
        print(f"Error extracting text: {e}")
        return
    
    # Initialize JSON file
    all_items = []
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # If file already exists, load it to continue
    if output_path.exists():
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                # If it already has image_ref, it means it passed through both passes
                if existing_data and "image_ref" in existing_data[0]:
                    print("[INFO] File already fully processed. Delete the file to reprocess.")
                    return
                all_items = existing_data
            print(f"[INFO] Continuing from {len(all_items)} items already extracted")
        except:
            all_items = []
    
    # PASS 1: Text only - Extract products from each page
    print("\n" + "=" * 60)
    print("PASS 1: Text extraction (identify products by page)")
    print("=" * 60)
    
    for page_data in pages_text:
        page_num = page_data["page"]
        page_text = page_data["text"]
        
        if not page_text.strip():
            print(f"\n[PASS 1] Page {page_num + 1}: No text, skipping...")
            continue
        
        print(f"\n[PASS 1] Processing page {page_num + 1}/{len(pages_text)}...")
        
        result = call_gpt_pdf_text_extraction(
            page_data=page_data,
            page_num=page_num,
            model=model
        )
        
        page_items = []
        for item in result.get("items", []):
            # Validate that it has a price
            if not item.get("price_raw") or item.get("price_raw", "").lower() in ["unknown", "none", ""]:
                print(f"  ⚠️  Item without valid price, skipping: {item.get('description', 'N/A')[:50]}")
                continue
            
            # Add page information
            item["source"] = "pdf"
            item["pdf_path"] = pdf_path
            item["page"] = page_num
            
            # Candidate images are all images from this page
            page_images = images_by_page.get(page_num, [])
            item["candidate_image_ids"] = [img.image_id for img in page_images]
            
            page_items.append(item)
            all_items.append(item)
        
        # Save incrementally after each page
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        
        print(f"  ✓ Items extracted on this page: {len(page_items)}")
        print(f"  ✓ Total accumulated: {len(all_items)} items")
        
        # Delay between pages
        sleep(2)
    
    print(f"\n[PASS 1 COMPLETED] Total items extracted: {len(all_items)}")
    
    # PASS 2: Image validation
    print("\n" + "=" * 60)
    print("PASS 2: Image validation (multimodal)")
    print("=" * 60)
    
    # Create dictionary of images by ID for quick access
    images_by_id: Dict[str, PdfImageRecord] = {img.image_id: img for img in image_records}
    
    for idx, item in enumerate(all_items):
        print(f"\n[PASS 2] Processing item {idx+1}/{len(all_items)}: {item.get('description', 'N/A')[:50]}")
        
        candidate_image_ids = item.get("candidate_image_ids", [])
        
        if not candidate_image_ids:
            item["image_ref"] = None
            print(f"  → No candidate images, image_ref = null")
        else:
            # Build candidate image paths
            candidate_paths = {}
            for img_id in candidate_image_ids:
                if img_id in images_by_id:
                    img_record = images_by_id[img_id]
                    if Path(img_record.path).exists():
                        candidate_paths[img_id] = img_record.path
            
            if not candidate_paths:
                item["image_ref"] = None
                print(f"  → Candidate images not found, image_ref = null")
            else:
                print(f"  → Validating {len(candidate_paths)} candidate images...")
                
                image_ref = call_gpt_image_validation(
                    description=item["description"],
                    price_raw=item["price_raw"],
                    candidate_image_paths=candidate_paths,
                    model=model
                )
                
                item["image_ref"] = image_ref
                if image_ref:
                    print(f"  ✓ Image assigned: {image_ref}")
                else:
                    print(f"  → No image matches, image_ref = null")
        
        # Save incrementally after each item
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        
        # Delay between items
        if idx < len(all_items) - 1:
            sleep(2)
    
    # Clean temporary field
    for item in all_items:
        if "candidate_image_ids" in item:
            del item["candidate_image_ids"]
    
    # Final save
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Process completed. Saved to {output_json}")
    print(f"[OK] Total final items: {len(all_items)}")

# ======================================
# MAIN
# ======================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Product extractor (WhatsApp + PDFs) with GPT multimodal - 2 passes.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    # WhatsApp subcommand
    p_wa = subparsers.add_parser("whatsapp", help="Process WhatsApp chat")
    p_wa.add_argument("--chat", required=True, help="Path to exported .txt chat file")
    p_wa.add_argument("--media_dir", required=True, help="Folder containing attached images")
    p_wa.add_argument("--out", default="whatsapp_items.json", help="Output JSON file")
    p_wa.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini, options: gpt-4o-mini, gpt-4o)")
    
    # PDF subcommand
    p_pdf = subparsers.add_parser("pdf", help="Process PDF(s) with products")
    p_pdf.add_argument("--pdf", help="Path to a specific PDF (or use --folder for folder)")
    p_pdf.add_argument("--folder", help="Folder with multiple PDFs to process")
    p_pdf.add_argument("--images_dir", default="pdf_images", help="Folder to save extracted images")
    p_pdf.add_argument("--out", default="pdf_items.json", help="Output JSON file")
    p_pdf.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use (default: gpt-4o-mini, options: gpt-4o-mini, gpt-4o)")
    
    args = parser.parse_args()
    
    if args.mode == "whatsapp":
        extract_from_whatsapp(
            chat_path=args.chat,
            media_dir=args.media_dir,
            output_json=args.out,
            model=args.model
        )
    elif args.mode == "pdf":
        if args.folder:
            # Process all PDFs in the folder
            pdf_folder = Path(args.folder)
            if not pdf_folder.exists():
                print(f"Error: Folder {args.folder} does not exist.")
                exit(1)
            
            pdf_files = list(pdf_folder.glob("*.pdf"))
            if not pdf_files:
                print(f"No PDF files found in {args.folder}")
                exit(1)
            
            print(f"Found {len(pdf_files)} PDF files to process")
            print("=" * 60)
            
            # Load existing items if file already exists
            all_items = []
            output_path = Path(args.out)
            if output_path.exists():
                try:
                    with open(args.out, "r", encoding="utf-8") as f:
                        all_items = json.load(f)
                    print(f"[INFO] Loaded {len(all_items)} existing items")
                except:
                    all_items = []
            
            # Process each PDF
            for idx, pdf_file in enumerate(pdf_files, 1):
                print(f"\n{'='*60}")
                print(f"Processing PDF {idx}/{len(pdf_files)}: {pdf_file.name}")
                print(f"{'='*60}")
                
                # Create specific image folder for this PDF
                pdf_images_dir = Path(args.images_dir) / pdf_file.stem
                
                # Process this PDF (will add items to existing JSON)
                extract_from_pdf(
                    pdf_path=str(pdf_file),
                    images_dir=str(pdf_images_dir),
                    output_json=args.out,
                    model=args.model
                )
            
            print(f"\n{'='*60}")
            print(f"Processing completed. Total items: {len(all_items)}")
            print(f"{'='*60}")
            
        elif args.pdf:
            # Process a specific PDF
            extract_from_pdf(
                pdf_path=args.pdf,
                images_dir=args.images_dir,
                output_json=args.out,
                model=args.model
            )
        else:
            print("Error: You must specify --pdf (a file) or --folder (folder with PDFs)")
            p_pdf.print_help()
    else:
        parser.print_help()
