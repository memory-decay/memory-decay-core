"""Upload final report to Google Docs using gws CLI."""

import json
import subprocess
import sys

DOC_ID = "1jIg9W2H6AIvMKTWNO9iO_i8bYg-CNmxZDMasLHkCe1s"

def gws_batch_update(requests):
    """Send batchUpdate to Google Docs."""
    cmd = [
        "gws", "docs", "documents", "batchUpdate",
        "--params", f'{{"documentId": "{DOC_ID}"}}',
        "--json", json.dumps({"requests": requests})
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
    return result.stdout

def make_requests_from_md(md_path):
    """Parse markdown and create Google Docs API requests."""
    with open(md_path) as f:
        lines = f.readlines()
    
    requests = []
    
    # Track current position
    # We'll build requests as insertText at document end
    
    # First, we need to clear the default newline
    # Then insert all content
    
    # Strategy: build the full text, then insert it,
    # then apply formatting with updateTextStyle
    
    # Parse the markdown into structured blocks
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')
        
        if line.startswith('# '):
            blocks.append(('HEADING_1', line[2:].strip()))
        elif line.startswith('## '):
            blocks.append(('HEADING_2', line[3:].strip()))
        elif line.startswith('### '):
            blocks.append(('HEADING_3', line[4:].strip()))
        elif line.startswith('#### '):
            blocks.append(('HEADING_4', line[5:].strip()))
        elif line.startswith('---'):
            blocks.append(('HR', ''))
        elif line.startswith('$$'):
            # Math block - collect all lines until $$
            math_lines = [line[2:]]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('$$'):
                math_lines.append(lines[i].rstrip('\n'))
                i += 1
            if i < len(lines):
                math_lines.append(lines[i].strip().rstrip('$'))
            blocks.append(('MATH', '\n'.join(math_lines)))
        elif line.startswith('|'):
            # Table - collect all table lines
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].rstrip('\n'))
                i += 1
            blocks.append(('TABLE', table_lines))
            continue  # already incremented i
        elif line.strip() == '':
            blocks.append(('EMPTY', ''))
        elif line.startswith('- '):
            # Collect list items
            list_items = []
            while i < len(lines) and lines[i].strip().startswith('- '):
                item = lines[i].strip()[2:]
                # Remove bold markers
                list_items.append(item)
                i += 1
            blocks.append(('LIST', list_items))
            continue
        elif line.startswith('1. ') or line.startswith('2. ') or line.startswith('3. '):
            # Numbered list
            list_items = []
            while i < len(lines) and lines[i].strip() and lines[i].strip()[0].isdigit() and '. ' in lines[i].strip()[:3]:
                item = lines[i].strip().split('. ', 1)[1]
                list_items.append(item)
                i += 1
            blocks.append(('NUMBERED_LIST', list_items))
            continue
        elif line.startswith('!['):
            # Image - skip for now
            blocks.append(('IMAGE', line))
        elif line.startswith('[') and '](' in line:
            blocks.append(('LINK', line))
        else:
            blocks.append(('TEXT', line))
        
        i += 1
    
    return blocks

def build_full_text(blocks):
    """Build the plain text content."""
    parts = []
    for btype, content in blocks:
        if btype == 'EMPTY':
            parts.append('')
        elif btype == 'HR':
            parts.append('─' * 40)
        elif btype == 'TABLE':
            for tl in content:
                parts.append(tl)
        elif btype == 'LIST':
            for item in content:
                parts.append(f'  • {item}')
        elif btype == 'NUMBERED_LIST':
            for idx, item in enumerate(content, 1):
                parts.append(f'  {idx}. {item}')
        elif btype == 'MATH':
            parts.append(content)
        elif btype == 'IMAGE':
            parts.append(f'[그림: {content}]')
        elif btype == 'LINK':
            parts.append(content)
        elif btype in ('HEADING_1', 'HEADING_2', 'HEADING_3', 'HEADING_4'):
            parts.append(content)
        else:
            parts.append(content)
    
    return '\n'.join(parts) + '\n'

def main():
    md_path = "docs/final-report.md"
    blocks = make_requests_from_md(md_path)
    
    # Build text content
    full_text = build_full_text(blocks)
    
    # Delete existing content and insert new
    requests = [
        {
            "insertText": {
                "location": {"index": 1},
                "text": full_text
            }
        }
    ]
    
    # Apply heading styles
    idx = 1
    for btype, content in blocks:
        if btype in ('HEADING_1', 'HEADING_2', 'HEADING_3', 'HEADING_4'):
            end_idx = idx + len(content) + 1  # +1 for newline
            requests.append({
                "updateParagraphStyle": {
                    "range": {"startIndex": idx, "endIndex": end_idx},
                    "paragraphStyle": {
                        "namedStyleType": btype
                    },
                    "fields": "namedStyleType"
                }
            })
        
        # Advance index
        if btype == 'TABLE':
            for tl in content:
                idx += len(tl) + 1
        elif btype == 'LIST':
            for item in content:
                idx += len(item) + 4 + 1  # "  • " + newline
        elif btype == 'NUMBERED_LIST':
            for num, item in enumerate(content, 1):
                idx += len(f'  {num}. {item}') + 1
        elif btype == 'EMPTY':
            idx += 1
        elif btype == 'HR':
            idx += 41  # "─" * 40 + newline
        else:
            text_len = len(str(content)) + 1
            idx += text_len
    
    # Split into batches of 50 (API limit)
    batch_size = 50
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        print(f"Sending batch {i//batch_size + 1}/{(len(requests)-1)//batch_size + 1} ({len(batch)} requests)...")
        result = gws_batch_update(batch)
        if result:
            resp = json.loads(result)
            if 'error' in resp:
                print(f"Error in batch: {resp['error']}")
                return
        import time
        time.sleep(0.5)
    
    print(f"Done! Document: https://docs.google.com/document/d/{DOC_ID}/edit")

if __name__ == "__main__":
    main()
