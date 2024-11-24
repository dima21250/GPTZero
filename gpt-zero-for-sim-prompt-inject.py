
import warnings

# Ignore FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import numpy as np
import pandas as pd
from model import GPT2PPL
from pathlib import Path
from datetime import datetime

# input_dir = Path("/Users/dima/Data/phd/main")
in_dir = Path("/mnt/isgnas/home/dima/Data/vader-llm/sim-prompt-inject/input")
in_files = [f.name for f in (in_dir.glob("*_vader.jsonl"))]
out_dir = Path("/mnt/isgnas/home/dima/Data/vader-llm/sim-prompt-inject/output")

llm = GPT2PPL()

orig_columns = [
    'generated_text',
    'model',
    'temperature',
    'max_tokens',
    'top_p',
    'frequency_penalty',
    'presence_penalty',
    'stop',
    'negative',
    'neutral',
    'positive',
    'compound'
]

new_columns = [
    'ppl',
    'ppl_line',
    'burst',
    'label'
]

keys = {'Burstiness', 'Perplexity', 'Perplexity per line', 'label'}

columns = orig_columns + new_columns

for file in in_files:
    print(file)
    out_path = out_dir / file
    # Adding a datetime to output file name
    out_format = str(out_path.parent / out_path.stem) + "_{}" + out_path.suffix
    out_path_gz = Path(out_format.format("gpt-zero"))

    df = pd.read_json(in_dir / file, lines=True)
    N = len(df)

    start = 0
    end = N

    df_gzero = pd.DataFrame(columns=columns)

    for i in range(start, end):
        print(i)
        row = df[orig_columns].iloc[i]
        text = row['generated_text']
        model = row['model']
        temperature = row['temperature']
        max_tokens = row['max_tokens']
        top_p = row['top_p']
        frequency_penalty = row['frequency_penalty']
        presence_penalty = row['presence_penalty']
        stop = row['stop']
        negative = row['negative']
        neutral = row['neutral']
        positive = row['positive']
        compound = row['compound']

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

        new_row = pd.DataFrame({
            'generated_text': [text],
            'model': [model],
            'temperature': [temperature],
            'max_tokens': [max_tokens],
            'top_p': [top_p],
            'frequency_penalty': [frequency_penalty],
            'presence_penalty': [presence_penalty],
            'stop': [stop],
            'negative': [negative],
            'neutral': [neutral],
            'positive': [positive],
            'compound': [compound],
            'ppl': [ppl],
            'ppl_line': [ppl_line],
            'burst': [burst],
            'label': [label]
        })

        df_gzero = pd.concat([df_gzero, new_row], ignore_index=True)

    df_gzero.to_json(out_path_gz, orient='records', lines=True)

