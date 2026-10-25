"""GR00T N1.5: global + wrist images, absolute link6 pose and suction."""
import os
from importlib import import_module


# Training uses NVIDIA's checkout; export/quantization uses Qualcomm's copy.
_prefix = os.environ.get('EPISODE_GR00T_PREFIX', 'gr00t')
BaseDataConfig = import_module(f'{_prefix}.experiment.data_config').BaseDataConfig
ComposedModalityTransform = import_module(f'{_prefix}.data.transform.base').ComposedModalityTransform
ConcatTransform = import_module(f'{_prefix}.data.transform.concat').ConcatTransform
_state_action = import_module(f'{_prefix}.data.transform.state_action')
StateActionToTensor = _state_action.StateActionToTensor
StateActionTransform = _state_action.StateActionTransform
_video = import_module(f'{_prefix}.data.transform.video')
VideoToTensor, VideoResize, VideoToNumpy = _video.VideoToTensor, _video.VideoResize, _video.VideoToNumpy
GR00TTransform = import_module(f'{_prefix}.model.transforms').GR00TTransform


class EpisodePoseConfig(BaseDataConfig):
    video_keys = ['video.front', 'video.wrist']
    state_keys = ['state.ee_position', 'state.ee_rotation', 'state.suction']
    action_keys = ['action.ee_position', 'action.ee_rotation', 'action.suction']
    language_keys = ['annotation.human.task_description']
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self):
        return ComposedModalityTransform(transforms=[
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation='linear'),
            VideoToNumpy(apply_to=self.video_keys),
            StateActionToTensor(apply_to=self.state_keys+self.action_keys),
            StateActionTransform(
                apply_to=self.state_keys+self.action_keys,
                normalization_modes={
                    'state.ee_position': 'min_max', 'action.ee_position': 'min_max',
                    'state.suction': 'binary', 'action.suction': 'binary',
                },
            ),
            ConcatTransform(video_concat_order=self.video_keys,
                            state_concat_order=self.state_keys, action_concat_order=self.action_keys),
            GR00TTransform(state_horizon=1, action_horizon=16, max_state_dim=64, max_action_dim=32),
        ])
