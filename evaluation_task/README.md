## How to run
### vqa

#### inference
```python ./vqa/all_vqa.py -m {chameleon,emu3,pixtral} -md {zero-shot,prompt} -n {N} -o {out_dir}```

#### evaluation
```python ./vqa/eval/eval_vqa.py -f {full_path}/{file_name}.json```

### captioning
#### inference
```python ./caption/all_caption.py -m {chameleon,emu3,pixtral} -n {N} -o {out_dir}```

#### evaluation
```python ./vqa/eval/cider_metrics/eval_json.py -f {full_path}/{file_name}.json```

>[!WARNING]
>Use `full_path` in evaluation task!
