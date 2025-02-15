
import torch
import numpy as np
import pandas as pd
from model import GPT2PPL
from pathlib import Path
from datetime import datetime

# input_dir = Path("/Users/dima/Data/phd/main")
in_dir = Path("/mnt/isgnas/home/dima/bigtwin-b/Code/annotune-generate-data")
in_file = "llm-text-analysis-main-vader.jsonl"

in_files = """
synthetic-first-contact-plot-summaries-llama-3-2-1b-16f-20250213-225304.jsonl
synthetic-first-contact-plot-summaries-llama-3-2-1b-16f-20250214-013020.jsonl
synthetic-first-contact-plot-summaries-llama-3-2-1b-20250213-154118.jsonl
synthetic-first-contact-plot-summaries-llama-3-2-1b-20250213-183125.jsonl
synthetic-first-contact-plot-summaries-llama-3-2-3b-16f-20250214-013658.jsonl
synthetic-first-contact-plot-summaries-llama-3-2-3b-16f-20250214-080912.jsonl
synthetic-first-contact-plot-summaries-llama-3-2-3b-20250213-155419.jsonl
synthetic-first-contact-plot-summaries-llama-3-2-3b-20250213-201538.jsonl
""".strip().split()

for in_file in in_files:
    out_dir = in_dir
    out_file = "gpt-zero-" + in_file
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

    columns = ['source_index', 'id', 'model', 'text', 'ppl', 'ppl_line', 'burst', 'label']
    df_gzero = pd.DataFrame(columns=columns)

    for i in range(start, end):
        print(i)
        source_index = i
        id = i
        model = df["model"].iloc[i]
        text = df["summary"].iloc[i]
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
