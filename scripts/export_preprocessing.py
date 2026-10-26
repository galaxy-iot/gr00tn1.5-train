#!/usr/bin/env python3
"""Export checkpoint embeddings and pinned Eagle preprocessing assets without loading the policy."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = '4af2b622892f7dcb5aae5a3fb70bcb02dc217b96'
EMBEDDING_KEY = 'backbone.eagle_model.language_model.model.embed_tokens.weight'


def find_processor(explicit):
    if explicit:
        return explicit.resolve()
    candidates = [ROOT/'third_party/Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model']
    candidates += sorted((Path.home()/'.qaihm/external_repos/models/grootn15/gr00t').glob(
        '*/gr00t/gr00t/model/backbone/eagle2_hg_model'))
    for path in candidates:
        if not path.is_dir():
            continue
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if revision == REVISION:
            return path
    raise ValueError('Pinned Eagle processor not found; pass --processor-dir from the training repository')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--processor-dir', type=Path)
    args = p.parse_args()
    checkpoint, output = args.checkpoint.resolve(), args.output.resolve()
    if output.exists():
        p.error('Use a fresh --output directory')
    source = find_processor(args.processor_dir)
    required = ['config.json', 'processor_config.json', 'preprocessor_config.json',
                'tokenizer_config.json', 'vocab.json', 'merges.txt', 'chat_template.json',
                'processing_eagle2_5_vl.py', 'image_processing_eagle2_5_vl_fast.py']
    for name in required:
        if not (source/name).is_file():
            raise ValueError(f'Missing processor asset: {source/name}')
    for path in (checkpoint/'config.json', checkpoint/'experiment_cfg/metadata.json'):
        if not path.is_file():
            raise ValueError(f'Missing checkpoint file: {path}')
    import numpy as np
    from safetensors import safe_open

    index = checkpoint/'model.safetensors.index.json'
    shards = [checkpoint/json.loads(index.read_text())['weight_map'][EMBEDDING_KEY]] if index.is_file() else sorted(checkpoint.glob('*.safetensors'))
    embedding = None
    for shard in shards:
        with safe_open(shard, framework='pt', device='cpu') as tensors:
            if EMBEDDING_KEY in tensors.keys():
                embedding = tensors.get_tensor(EMBEDDING_KEY)
                break
    if embedding is None:
        raise ValueError(f'Checkpoint does not contain {EMBEDDING_KEY}')
    config = json.loads((source/'config.json').read_text())
    expected = (config['text_config']['vocab_size'], config['text_config']['hidden_size'])
    if tuple(embedding.shape) != expected:
        raise ValueError(f'Embedding/processor mismatch: {embedding.shape} vs {expected}')
    output.mkdir(parents=True)
    processor = output/'processor'
    processor.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.suffix in ('.json', '.txt', '.py', '.jinja', '.model'):
            shutil.copy2(path, processor/path.name)
    shutil.copytree(checkpoint/'experiment_cfg', output/'experiment_cfg')
    shutil.copy2(checkpoint/'config.json', output/'model_config.json')
    shutil.copy2(ROOT/'episode_config.py', output/'episode_config.py')
    shutil.copy2(ROOT/'scripts/prepare_language.py', output/'prepare_language.py')
    # Convert rows in chunks to avoid holding an additional full float32 table in RAM.
    table = np.lib.format.open_memmap(output/'embedding.npy', mode='w+', dtype=np.float32, shape=expected)
    for start in range(0, expected[0], 4096):
        table[start:start+4096] = embedding[start:start+4096].float().numpy()
    table.flush()
    del table
    raw_tokens = (config['vision_config']['image_size']//config['vision_config']['patch_size'])**2
    vision_tokens = int(raw_tokens*config['downsample_ratio']**2) if config.get('use_pixel_shuffle') else raw_tokens
    settings = {
        'checkpoint': str(checkpoint), 'embedding_key': EMBEDDING_KEY,
        'embedding_shape': expected, 'embedding_dtype': 'float32',
        'processor_source': str(source), 'training_repository_revision': REVISION,
        'image_token_index': config['image_token_index'],
        'camera_order': ['front', 'wrist'], 'image_size': [224, 224],
        'vision_tokens_per_camera': vision_tokens, 'language_token_budget': 128,
        'vlm_seq_len': 128+2*vision_tokens, 'padding_side': 'left',
        'sequence_padding': 'right-pad processor output to vlm_seq_len, matching GrootApp',
        'dependencies': {name: importlib.metadata.version(name) for name in
                         ('torch', 'torchvision', 'transformers', 'tokenizers', 'numpy', 'pillow', 'safetensors')},
        'processor_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in processor.iterdir()},
        'llm_mask_min_value': -100.0,
        'scope': 'Language preprocessing resources. Image/state processing and full action inference are separate.',
    }
    (output/'preprocess_config.json').write_text(json.dumps(settings, indent=2, ensure_ascii=False)+'\n')
    (output/'requirements-preprocessing.txt').write_text(''.join(
        f'{name}=={version}\n' for name, version in settings['dependencies'].items()
        if name != 'safetensors')+'requests\npackaging\n')
    (output/'README.md').write_text(
        '# GR00T language preprocessing bundle\n\n'
        'Install requirements-preprocessing.txt in a compatible Python environment.\n'
        'Run: python prepare_language.py --bundle . --instruction "Pick up the object." --output inputs.npz\n\n'
        'The NPZ provides text embeddings, image slot mask and the LLM causal mask.\n'
        'Replace input_embeds[image_token_mask] with ViT output before calling the LLM.\n'
        'Do not pass these language tensors as a complete robot inference input.\n'
        'Images, normalized robot state, projector/DiT masks, diffusion scheduling and action inverse transforms are separate.\n'
        'See preprocess_config.json for provenance and camera/sequence settings.\n')
    # Completion receipt is written after validating an offline Processor reload and token lookup.
    os.environ['HF_MODULES_CACHE'] = str(output/'.hf_modules')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from prepare_language import prepare_language
    sample = prepare_language(output, 'Pick up the object.')
    np.savez(output/'language_example.npz', **sample)
    (output/'export_report.json').write_text(json.dumps({'complete': True,
        'offline_processor_reload_checked': True, 'example_instruction': 'Pick up the object.',
        'example_shapes': {key: list(value.shape) for key, value in sample.items()}}, indent=2)+'\n')
    print(f'Exported preprocessing resources: {output}', flush=True)


if __name__ == '__main__':
    main()
