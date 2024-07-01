import os
import json 
import nest_asyncio
from llama_parse import LlamaParse

nest_asyncio.apply()
os.environ["LLAMA_CLOUD_API_KEY"] = ""

path = "/Users/a11492/Desktop/develop/lcl/5G이용약관.pdf"
vanilaParsing = LlamaParse(result_type="markdown").load_data(path)

chunk_by_page = vanilaParsing[0].text.split('\n---\n#')

all_paragraph = []
for page_idx, page in enumerate(chunk_by_page[1:]):
  if page_idx == 0 or page_idx == 211:
    pass
  chunk_by_paragraph = page.split('# ')
  
  for pgph in chunk_by_paragraph:
    chunk = []
    pgph = pgph.strip()
    if not pgph == '':
      chunk.append(pgph)

    chunk_str = ''.join(chunk)
    if not chunk_str == '': 
      all_paragraph.append(chunk_str)
  
with open('skt_이용약관.jsonl','w',encoding='utf-8') as _f:
    for item in all_paragraph:
        _f.write(json.dumps(item, ensure_ascii=False)+"\n")