#!/usr/bin/env python3
"""Local float ONNX export via Qualcomm's N1.5 adapters; no cloud submission."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def register_config():
    # Qualcomm imports the pinned repository through its external-repo package.
    from qai_hub_models.models.grootn15.external_repos.gr00t.gr00t.experiment.data_config import DATA_CONFIG_MAP
    os.environ['EPISODE_GR00T_PREFIX'] = 'qai_hub_models.models.grootn15.external_repos.gr00t.gr00t'
    sys.path.insert(0, str(ROOT))
    from episode_config import EpisodePoseConfig
    DATA_CONFIG_MAP['episode_pose'] = EpisodePoseConfig()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component', choices=['all', 'vit', 'llm', 'vlm_proj', 'dit'], default='all')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint/'config.json').exists() or not (checkpoint/'experiment_cfg').is_dir():
        raise ValueError('Provide a complete trained checkpoint containing config.json and experiment_cfg/')
    report = json.loads((args.dataset/'build_report.json').read_text())
    if not report['ready_for_training']:
        print('Dataset annotations are unconfirmed; continuing ONNX export.', flush=True)
    # Heavy imports occur only after validating paths and --help.
    import onnx
    import torch
    from qai_hub_models.models.grootn15.model import GrootCollection
    from qai_hub_models.utils.input_spec import make_torch_inputs
    from qai_hub_models.utils.qai_hub_helpers import export_torch_to_onnx_zip
    register_config()
    args.output.mkdir(parents=True, exist_ok=False)
    collection = GrootCollection.from_pretrained(checkpoint=str(checkpoint), data_config='episode_pose',
                                                embodiment_tag='new_embodiment', host_device='cpu')
    names = list(collection.components) if args.component == 'all' else [args.component]
    exported = {}
    for name in names:
        model = collection.components[name].eval().cpu()
        spec = model.get_input_spec()
        output = args.output/name
        with torch.inference_mode():
            # Directory output keeps large external weights beside the ONNX graph.
            artifact = Path(export_torch_to_onnx_zip(
                model, output, make_torch_inputs(spec), input_names=list(spec),
                output_names=model.get_output_names(), skip_zip=True,
                torch_export_kwargs={'opset_version': 20}, prefer_external_weight=True))
        graphs = list(artifact.rglob('*.onnx')) if artifact.is_dir() else [artifact]
        if not graphs:
            raise RuntimeError(f'No ONNX graph found: {artifact}')
        for graph in graphs:
            onnx.checker.check_model(str(graph))
        exported[name] = {'artifact': str(artifact.resolve()), 'inputs': list(spec),
                          'outputs': model.get_output_names(), 'onnx_structure_checked': True,
                          'numerical_equivalence_checked': False}
        print(f'Exported {name}: {artifact}', flush=True)
    shutil.copytree(checkpoint/'experiment_cfg', args.output/'experiment_cfg')
    shutil.copyfile(ROOT/'episode_config.py', args.output/'episode_config.py')
    shutil.copyfile(args.dataset/'robot.urdf', args.output/'robot.urdf')
    (args.output/'export_report.json').write_text(json.dumps({
        'checkpoint': str(checkpoint), 'embodiment': 'new_embodiment', 'data_config': 'episode_pose',
        'components': exported, 'precision': 'float32', 'npu_compiled': False,
        'note': 'Keep external weights. Requires GrootApp preprocessing/scheduling and action inverse transforms.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
