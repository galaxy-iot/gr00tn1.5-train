#!/usr/bin/env python3
"""Offline text -> checkpoint embeddings for the split GR00T ONNX pipeline.

Image slots are placeholders: the caller must replace them with current ViT outputs.
This utility intentionally does not produce robot state or run the action model.
"""
import argparse
import json
import os
from pathlib import Path


def prepare_language(bundle, instruction):
    if not instruction.strip():
        raise ValueError('Instruction must be nonempty')
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    bundle = Path(bundle).resolve()
    cfg = json.loads((bundle/'preprocess_config.json').read_text())
    processor = AutoProcessor.from_pretrained(str(bundle/'processor'), trust_remote_code=True,
                                             local_files_only=True, use_fast=True)
    processor.tokenizer.padding_side = cfg['padding_side']
    # Exact conversation layout used by GR00TTransform._apply_vlm_processing.
    # Image content is irrelevant for text IDs; fixed-size images establish the camera token slots.
    images = [Image.new('RGB', tuple(cfg['image_size'])) for _ in cfg['camera_order']]
    conversation = [{'role': 'user', 'content':
        [{'type': 'image', 'image': image} for image in images]
        + [{'type': 'text', 'text': instruction}]}]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = processor.process_vision_info(conversation)
    inputs = processor(text=[text], images=image_inputs, return_tensors='pt', padding=True)
    ids = inputs['input_ids'].numpy()
    mask = inputs['attention_mask'].numpy().astype(np.int32)
    length = cfg['vlm_seq_len']
    if ids.shape[1] > length:
        raise ValueError(f'Instruction exceeds exported sequence length: {ids.shape[1]} > {length}')
    padding = length-ids.shape[1]
    ids = np.pad(ids, ((0, 0), (0, padding)), constant_values=processor.tokenizer.pad_token_id)
    mask = np.pad(mask, ((0, 0), (0, padding)))
    weights = np.load(bundle/'embedding.npy', mmap_mode='r')
    embeds = weights[ids]
    selected = ids == cfg['image_token_index']
    if int(selected.sum()) != len(images)*cfg['vision_tokens_per_camera']:
        raise ValueError('Processor image token count differs from exported ViT layout')
    # Same float32 additive causal mask as preprocess_llm (single sample).
    minimum = np.float32(-100.0)
    causal = np.triu(np.full((length, length), minimum, dtype=np.float32), k=1)[None, None]
    causal = np.where((causal == 0) & (mask[:, None, None, :] == 0), minimum, causal)
    return {'input_ids': ids, 'attention_mask': mask, 'input_embeds': embeds,
            'image_token_mask': selected, 'llm_attention_mask': causal}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bundle', type=Path)
    mode.add_argument('--input-ids', type=Path, help='Processor-generated integer .npy [batch, sequence]')
    p.add_argument('--embedding', type=Path, default=Path(__file__).parent/'embedding.npy')
    p.add_argument('--instruction')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh --output file')
    if args.input_ids is not None:
        import numpy as np
        ids = np.load(args.input_ids, allow_pickle=False)
        table = np.load(args.embedding, mmap_mode='r', allow_pickle=False)
        if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer) or ids.size == 0:
            p.error('input IDs must be a nonempty integer [batch, sequence] array')
        if table.ndim != 2 or ids.min() < 0 or ids.max() >= len(table):
            p.error('Token IDs exceed the embedding vocabulary, or table is invalid')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as handle:
            np.save(handle, table[ids], allow_pickle=False)
        return
    if not args.instruction:
        p.error('--instruction is required with --bundle')
    # Set before importing transformers, so no global HF cache is required.
    os.environ.setdefault('HF_MODULES_CACHE', str(args.bundle.resolve()/'.hf_modules'))
    import numpy as np
    inputs = prepare_language(args.bundle, args.instruction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as handle:
        np.savez(handle, **inputs)
    print(json.dumps({key: list(value.shape) for key, value in inputs.items()}))


if __name__ == '__main__':
    main()
