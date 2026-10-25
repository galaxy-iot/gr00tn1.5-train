"""Load component dumps as complete model invocations, preserving camera groups."""


def load_samples(sink, component_name, spec, io_type, num_samples, *, num_cameras):
    import numpy as np

    if component_name != 'vit':
        arrays = sink.load(component_name, spec, io_type=io_type, num_samples=num_samples)
        return {name: [array[i:i + 1] for i in range(num_samples)]
                for name, array in arrays.items()}

    # Upstream dump strips axis 0 and writes each camera to a separate file.
    # Reassemble consecutive cameras from the same robot step for BOTH inputs
    # and reference outputs. Keep samples separate instead of flattening them.
    base = sink._io_dir(component_name, io_type)
    result = {}
    for name, tensor_spec in spec.items():
        samples = []
        for i in range(num_samples):
            paths = [base / f'{name}_{i * num_cameras + camera:04d}.npy'
                     for camera in range(num_cameras)]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f'Missing camera sample: {path}; regenerate calibration data')
            array = np.stack([np.load(path, allow_pickle=False) for path in paths])
            if tensor_spec is not None:
                shape, dtype = tensor_spec
                array = array.astype(dtype, copy=False)
                if array.shape != tuple(shape):
                    raise ValueError(f'{component_name}.{name} sample {i}: '
                                     f'expected {shape}, got {array.shape}')
            samples.append(array)
        result[name] = samples
    return result


def evaluate_component(recipe, name, component, sink, count, num_cameras):
    import numpy as np
    import torch

    inputs = load_samples(sink, name, component.get_input_spec(),
                          recipe.ComponentIOType.INPUTS, count, num_cameras=num_cameras)
    output_names = component.get_output_names()
    expected = load_samples(sink, name, dict.fromkeys(output_names),
                            recipe.ComponentIOType.OUTPUTS, count, num_cameras=num_cameras)
    actual = {key: [] for key in output_names}
    with torch.no_grad():
        for i in range(count):
            outputs = component.forward(*(torch.from_numpy(values[i]) for values in inputs.values()))
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            for key, value in zip(output_names, outputs, strict=True):
                array = value.detach().cpu().numpy()
                if array.shape != expected[key][i].shape:
                    raise ValueError(f'{name}.{key}: output {array.shape} differs from '
                                     f'reference {expected[key][i].shape}')
                actual[key].append(array)
    print(f'\n[calib_eval] {name}')
    for key in output_names:
        reference = np.concatenate(expected[key], axis=0)
        result = np.concatenate(actual[key], axis=0)
        sqnr, _, _ = recipe.get_sqnr(reference, result)
        cosine = recipe.get_cossim(reference.ravel(), result.ravel())
        print(f'  {key}: SQNR={sqnr:.2f} dB; CosSim={cosine:.4f}')


def run_quantization(recipe, args):
    """Use the upstream models/quantizers with camera-aware sample batching."""
    import torch

    torch.cuda.set_device('cuda:0')
    policy = recipe.load_checkpoint(checkpoint=str(args.checkpoint.resolve()),
                                    data_config='episode_pose',
                                    embodiment_tag='new_embodiment', device='cuda:0')
    dataset = recipe.LeRobotSingleDataset(
        dataset_path=str((args.dataset / 'train').resolve()),
        modality_configs=policy.modality_config, video_backend='decord',
        video_backend_kwargs=None, transforms=None, embodiment_tag='new_embodiment')
    if len(dataset) < 2 * args.num_samples:
        raise ValueError(f'Need at least {2 * args.num_samples} dataset steps for separate '
                         f'calibration/evaluation splits; found {len(dataset)}')
    components = (recipe.GrootQuantizedCollection.component_class_names
                  if args.component == 'all' else [args.component])
    for name in components:
        print(f'\n[quantize] Quantizing component: {name}', flush=True)
        quant_args = recipe.COMPONENT_QUANT_ARGS[name]
        component = recipe.GrootQuantizedCollection.component_classes[name].from_pretrained(
            checkpoint=str(args.checkpoint.resolve()), host_device='cuda:0',
            precision=recipe.QUANT_PRECISION[name], data_config='episode_pose',
            embodiment_tag='new_embodiment', **quant_args)
        spec = component.get_input_spec()
        num_cameras = spec['pixel_values'][0][0] if name == 'vit' else 1
        # The upstream cache does not identify checkpoint/dataset. Always collect
        # fresh data so a previous run cannot silently supply mismatched samples.
        calib, evaluation = recipe.GrootApp.get_calibration_data(
            policy=policy, dataset=dataset, component_name=name,
            num_samples=args.num_samples, host_device='cuda:0', use_cache=False,
            rng_seed=42)
        entries = load_samples(calib, name, spec, recipe.ComponentIOType.INPUTS,
                               args.num_samples, num_cameras=num_cameras)
        loader = recipe.dataset_entries_to_dataloader(entries)
        component.quantize(loader, num_samples=args.num_samples, **quant_args)
        component.save_calibrated_checkpoint(output_checkpoint=str(args.output.resolve()))
        evaluate_component(recipe, name, component, evaluation, args.num_samples, num_cameras)
        calib.clear()
        evaluation.clear()
        del component, loader, entries
        torch.cuda.empty_cache()
    print('\n[quantize] Done.')
