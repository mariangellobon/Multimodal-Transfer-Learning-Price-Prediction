from pathlib import Path
import re
import requests
from urllib.parse import urlparse, parse_qs
import time

chat_path = Path("WhatsApp Chat - Sloan Buy _ Sell 26s + 25s (1)/_chat.txt")
output_dir = Path("output/downloaded_docs")
output_dir.mkdir(parents=True, exist_ok=True)

# Patterns to detect document links
doc_patterns = [
    r'https?://docs\.google\.com/[^\s]+',
    r'https?://drive\.google\.com/[^\s]+',
    r'https?://.*\.pdf',
]

def extract_doc_id(google_url):
    """Extracts the document ID from a Google Docs URL."""
    # For Google Docs/Presentations
    if '/presentation/d/' in google_url:
        match = re.search(r'/presentation/d/([a-zA-Z0-9_-]+)', google_url)
        if match:
            return match.group(1), 'presentation'
    elif '/document/d/' in google_url:
        match = re.search(r'/document/d/([a-zA-Z0-9_-]+)', google_url)
        if match:
            return match.group(1), 'document'
    elif '/spreadsheets/d/' in google_url:
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9_-]+)', google_url)
        if match:
            return match.group(1), 'spreadsheet'
    # For Google Drive
    elif '/file/d/' in google_url:
        match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', google_url)
        if match:
            return match.group(1), 'file'
    return None, None

def convert_google_docs_url(url, export_format='pdf'):
    """Converts a Google Docs URL to download format."""
    doc_id, doc_type = extract_doc_id(url)
    if not doc_id:
        return None
    
    # Mapping of types to export formats
    format_map = {
        'presentation': 'pptx',  # or 'pdf'
        'document': 'docx',  # or 'pdf'
        'spreadsheet': 'xlsx',  # or 'pdf'
        'file': 'pdf'
    }
    
    export_format = format_map.get(doc_type, 'pdf')
    
    if doc_type == 'presentation':
        return f"https://docs.google.com/presentation/d/{doc_id}/export/{export_format}"
    elif doc_type == 'document':
        return f"https://docs.google.com/document/d/{doc_id}/export?format={export_format}"
    elif doc_type == 'spreadsheet':
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format={export_format}"
    elif doc_type == 'file':
        return f"https://drive.google.com/uc?export=download&id={doc_id}"
    
    return None

def is_valid_pdf(content):
    """Verifies if the content is a valid PDF."""
    # A valid PDF must start with %PDF
    if len(content) < 4:
        return False
    return content[:4] == b'%PDF'

def download_file(url, output_path, max_retries=3):
    """Downloads a file from a URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
            if response.status_code == 200:
                content = response.content
                
                # Verify if it's a valid PDF
                if not is_valid_pdf(content):
                    # Verify if it's HTML (error page or login)
                    if content.startswith(b'<') or b'<html' in content[:1000].lower():
                        return False, "Downloaded HTML instead of PDF (probably requires authentication)"
                    return False, "Downloaded file is not a valid PDF"
                
                output_path.write_bytes(content)
                return True, f"Downloaded: {len(content)} bytes (valid PDF)"
            elif response.status_code == 403:
                return False, "Access denied (requires authentication)"
            elif response.status_code == 401:
                return False, "Unauthorized (requires authentication)"
            else:
                return False, f"HTTP error {response.status_code}"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
                continue
            return False, f"Connection error: {str(e)}"
    
    return False, "Error after multiple attempts"

def main():
    # Read chat and extract unique links
    chat_text = chat_path.read_text(encoding="utf-8")
    all_links = set()
    
    for pattern in doc_patterns:
        matches = re.finditer(pattern, chat_text, re.IGNORECASE)
        for match in matches:
            link = match.group(0).rstrip('.,;!?)')
            all_links.add(link)
    
    print(f"Unique links found: {len(all_links)}")
    
    downloaded = 0
    failed = 0
    skipped = 0
    
    for idx, link in enumerate(sorted(all_links), 1):
        print(f"\n[{idx}/{len(all_links)}] Processing: {link[:80]}...")
        
        # Determine filename
        if 'docs.google.com' in link:
            doc_id, doc_type = extract_doc_id(link)
            if doc_id:
                if doc_type == 'presentation':
                    filename = f"presentation_{doc_id}.pdf"
                    download_url = convert_google_docs_url(link, 'pdf')
                elif doc_type == 'document':
                    filename = f"document_{doc_id}.pdf"
                    download_url = convert_google_docs_url(link, 'pdf')
                elif doc_type == 'spreadsheet':
                    filename = f"spreadsheet_{doc_id}.pdf"
                    download_url = convert_google_docs_url(link, 'pdf')
                else:
                    filename = f"google_doc_{doc_id}.pdf"
                    download_url = convert_google_docs_url(link, 'pdf')
            else:
                print(f"  ⚠️  Could not extract document ID")
                skipped += 1
                continue
        elif 'drive.google.com' in link:
            doc_id, _ = extract_doc_id(link)
            if doc_id:
                filename = f"drive_file_{doc_id}.pdf"
                download_url = convert_google_docs_url(link, 'pdf')
            else:
                print(f"  ⚠️  Could not extract file ID")
                skipped += 1
                continue
        elif link.endswith('.pdf'):
            # Direct PDF
            parsed = urlparse(link)
            filename = Path(parsed.path).name or f"pdf_{idx}.pdf"
            download_url = link
        else:
            print(f"  ⚠️  Unsupported link type")
            skipped += 1
            continue
        
        if not download_url:
            print(f"  ⚠️  Could not generate download URL")
            skipped += 1
            continue
        
        output_path = output_dir / filename
        
        # Check if already exists
        if output_path.exists():
            print(f"  ✓ Already exists: {filename}")
            downloaded += 1
            continue
        
        # Try to download
        success, message = download_file(download_url, output_path)
        
        if success:
            print(f"  ✓ {message}")
            downloaded += 1
        else:
            print(f"  ✗ {message}")
            failed += 1
        
        # Small pause between downloads
        time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total processed: {len(all_links)}")
    print(f"\nFiles saved to: {output_dir}")

if __name__ == "__main__":
    main()

