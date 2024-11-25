
import torch
import numpy as np
import pandas as pd
from model import GPT2PPL
from pathlib import Path
from datetime import datetime

# input_dir = Path("/Users/dima/Data/phd/main")
in_dir = Path("/mnt/isgnas/home/dima/Data/vader-llm")
in_file = "text-reuters-21578.jsonl"

out_dir = Path("/mnt/isgnas/home/dima/Output/llm-change/baseline")
out_file = "gpt-zero-reuters.jsonl"
out_path = out_dir / out_file

now = datetime.now()
now_string = now.strftime("%Y%m%d-%H%M%S")

# Adding a datetime to output file name
out_format = str(out_path.parent / out_path.stem) + "-{}" + out_path.suffix
out_path_dt = Path(out_format.format(now_string))

llm = GPT2PPL()

df = pd.read_json(in_dir / in_file, lines=True)
N = len(df)

start = 0
end = N

columns = ['source_index', 'model', 'text', 'ppl', 'ppl_line', 'burst', 'label']
df_gzero = pd.DataFrame(columns=columns)

for i in range(start, end):
    print(i)
    source_index = df["source_index"].iloc[i]
    model = "reuters"
    text = df["text"].iloc[i]
    keys = {'Burstiness', 'Perplexity', 'Perplexity per line', 'label'}
    
    results = llm(text)
    if keys.difference(results[0].keys()) == set():
        ppl = results[0]['Perplexity']
        ppl_line = results[0]['Perplexity per line']
        burst = results[0]['Burstiness']
        label = results[0]['label']

        # Move off GPU if needed.
        ppl = ppl.cpu().numpy() if isinstance(ppl, torch.Tensor) and ppl.is_cuda else ppl
        ppl_line = ppl_line.cpu().numpy() if isinstance(ppl_line, torch.Tensor) and ppl_line.is_cuda else ppl_line
        burst = burst.cpu().numpy() if isinstance(burst, torch.Tensor) and burst.is_cuda else burst
        label = label.cpu().numpy() if isinstance(label, torch.Tensor) and label.is_cuda else label

        # Make it serializable
        ppl = float(ppl)
        ppl_line = float(ppl_line)
        burst = float(burst)
        label = float(label)

    else:
        ppl = ppl_line = burst = label = np.nan

    row = pd.DataFrame({
        'source_index' : [source_index], 
        'id' : [id], 
        'model' : [model], 
        'text' : [text], 
        'ppl' : [ppl], 
        'ppl_line' : [ppl_line], 
        'burst' : [burst], 
        'label' : [label]
    })

    df_gzero = pd.concat([df_gzero, row], ignore_index=True)

df_gzero.to_json(out_path_dt, orient='records', lines=True)
