import numpy as np
# import gradio as gr
import torch
import os
import warnings
from gradio.processing_utils import convert_to_16_bit_wav
from typing import Dict, List, Optional, Union

import utils
from infer import get_net_g, infer
from models import SynthesizerTrn
from models_jp_extra import SynthesizerTrn as SynthesizerTrnJPExtra

# from .log import logger
from .constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
)


class Model:
    def __init__(
        self, model_path: str, config_path: str, style_vec_path: str, device: str
    ):
        self.model_path: str = model_path
        self.config_path: str = config_path
        self.device: str = device
        self.style_vec_path: str = style_vec_path
        self.hps: utils.HParams = utils.get_hparams_from_file(self.config_path)
        self.spk2id: Dict[str, int] = self.hps.data.spk2id
        self.id2spk: Dict[int, str] = {v: k for k, v in self.spk2id.items()}

        self.num_styles: int = self.hps.data.num_styles
        if hasattr(self.hps.data, "style2id"):
            self.style2id: Dict[str, int] = self.hps.data.style2id
        else:
            self.style2id: Dict[str, int] = {str(i): i for i in range(self.num_styles)}
        if len(self.style2id) != self.num_styles:
            raise ValueError(
                f"Number of styles ({self.num_styles}) does not match the number of style2id ({len(self.style2id)})"
            )

        self.style_vectors: np.ndarray = np.load(self.style_vec_path)
        if self.style_vectors.shape[0] != self.num_styles:
            raise ValueError(
                f"The number of styles ({self.num_styles}) does not match the number of style vectors ({self.style_vectors.shape[0]})"
            )

        self.net_g: Union[SynthesizerTrn, SynthesizerTrnJPExtra, None] = None

    def load_net_g(self):
        self.net_g = get_net_g(
            model_path=self.model_path,
            version=self.hps.version,
            device=self.device,
            hps=self.hps,
        )

    def get_style_vector(self, style_id: int, weight: float = 1.0) -> np.ndarray:
        mean = self.style_vectors[0]
        style_vec = self.style_vectors[style_id]
        style_vec = mean + (style_vec - mean) * weight
        return style_vec

    def get_style_vector_from_audio(
        self, audio_path: str, weight: float = 1.0
    ) -> np.ndarray:
        from style_gen import get_style_vector

        xvec = get_style_vector(audio_path)
        mean = self.style_vectors[0]
        xvec = mean + (xvec - mean) * weight
        return xvec

    def infer(
        self,
        text: str,
        language: str = "JP",
        sid: int = 0,
        reference_audio_path: Optional[str] = None,
        sdp_ratio: float = DEFAULT_SDP_RATIO,
        noise: float = DEFAULT_NOISE,
        noisew: float = DEFAULT_NOISEW,
        length: float = DEFAULT_LENGTH,
        line_split: bool = DEFAULT_LINE_SPLIT,
        split_interval: float = DEFAULT_SPLIT_INTERVAL,
        assist_text: Optional[str] = None,
        assist_text_weight: float = DEFAULT_ASSIST_TEXT_WEIGHT,
        use_assist_text: bool = False,
        style: str = DEFAULT_STYLE,
        style_weight: float = DEFAULT_STYLE_WEIGHT,
        given_tone: Optional[list[int]] = None,
    ) -> tuple[int, np.ndarray]:
        print(f"Start generating audio data from text:\n{text}")
        if language != "JP" and self.hps.version.endswith("JP-Extra"):
            raise ValueError(
                "The model is trained with JP-Extra, but the language is not JP"
            )
        if reference_audio_path == "":
            reference_audio_path = None
        if assist_text == "" or not use_assist_text:
            assist_text = None

        if self.net_g is None:
            self.load_net_g()
        if reference_audio_path is None:
            style_id = self.style2id[style]
            style_vector = self.get_style_vector(style_id, style_weight)
        else:
            style_vector = self.get_style_vector_from_audio(
                reference_audio_path, style_weight
            )
        if not line_split:
            with torch.no_grad():
                audio = infer(
                    text=text,
                    sdp_ratio=sdp_ratio,
                    noise_scale=noise,
                    noise_scale_w=noisew,
                    length_scale=length,
                    sid=sid,
                    language=language,
                    hps=self.hps,
                    net_g=self.net_g,
                    device=self.device,
                    assist_text=assist_text,
                    assist_text_weight=assist_text_weight,
                    style_vec=style_vector,
                    given_tone=given_tone,
                )
        else:
            texts = text.split("\n")
            texts = [t for t in texts if t != ""]
            audios = []
            with torch.no_grad():
                for i, t in enumerate(texts):
                    audios.append(
                        infer(
                            text=t,
                            sdp_ratio=sdp_ratio,
                            noise_scale=noise,
                            noise_scale_w=noisew,
                            length_scale=length,
                            sid=sid,
                            language=language,
                            hps=self.hps,
                            net_g=self.net_g,
                            device=self.device,
                            assist_text=assist_text,
                            assist_text_weight=assist_text_weight,
                            style_vec=style_vector,
                        )
                    )
                    if i != len(texts) - 1:
                        audios.append(np.zeros(int(44100 * split_interval)))
                audio = np.concatenate(audios)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio = convert_to_16_bit_wav(audio)
        print("Audio data generated successfully")
        return (self.hps.data.sampling_rate, audio)


class ModelHolder:
    def __init__(self, root_dir: str, device: str):
        self.root_dir: str = root_dir
        self.device: str = device
        self.model_files_dict: Dict[str, List[str]] = {}
        self.current_model: Optional[Model] = None
        self.model_names: List[str] = []
        self.models: List[Model] = []
        self.refresh()

    def refresh(self):
        self.model_files_dict = {}
        self.model_names = []
        self.current_model = None
        model_dirs = [
            d
            for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        for model_name in model_dirs:
            model_dir = os.path.join(self.root_dir, model_name)
            model_files = [
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.endswith(".pth") or f.endswith(".pt") or f.endswith(".safetensors")
            ]
            if len(model_files) == 0:
                print(
                    f"No model files found in {self.root_dir}/{model_name}, so skip it"
                )
                continue
            self.model_files_dict[model_name] = model_files
            self.model_names.append(model_name)



    def load_model(self , model_name , model_path):

            self.current_model = Model(
                model_path=model_path,
                config_path=os.path.join(self.root_dir, model_name, "config.json"),
                style_vec_path=os.path.join(self.root_dir, model_name, "style_vectors.npy"),
                device=self.device,
            )
            speakers = list(self.current_model.spk2id.keys())
            styles = list(self.current_model.style2id.keys())
            return styles[0] , speakers[0]

