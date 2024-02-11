import argparse
import glob
import json
import logging
import os
import re
import subprocess

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from scipy.io.wavfile import read

# from common.log import logger

MATPLOTLIB_FLAG = False


def download_checkpoint(
    dir_path, repo_config, token=None, regex="G_*.pth", mirror="openi"
):
    repo_id = repo_config["repo_id"]
    f_list = glob.glob(os.path.join(dir_path, regex))
    if f_list:
        print("Use existed model, skip downloading.")
        return
    for file in ["DUR_0.pth", "D_0.pth", "G_0.pth"]:
        hf_hub_download(repo_id, file, local_dir=dir_path, local_dir_use_symlinks=False)


def load_checkpoint(
    checkpoint_path, model, optimizer=None, skip_optimizer=False, for_infer=False
):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    print(
        f"Loading model and optimizer at iteration {iteration} from {checkpoint_path}"
    )
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    elif optimizer is None and not skip_optimizer:
        # else:      Disable this line if Infer and resume checkpoint,then enable the line upper
        new_opt_dict = optimizer.state_dict()
        new_opt_dict_params = new_opt_dict["param_groups"][0]["params"]
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "emb_g" not in k
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except:
            # For upgrading from the old version
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                print(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            elif "enc_q" in k and for_infer:
                continue
            else:
                print(f"{k} is not in the checkpoint {checkpoint_path}")

            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    print("Loaded '{}' (iteration {})".format(checkpoint_path, iteration))

    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    print(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def save_safetensors(model, iteration, checkpoint_path, is_half=False, for_infer=False):
    """
    Save model with safetensors.
    """
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    keys = []
    for k in state_dict:
        if "enc_q" in k and for_infer:
            continue  # noqa: E701
        keys.append(k)

    new_dict = (
        {k: state_dict[k].half() for k in keys}
        if is_half
        else {k: state_dict[k] for k in keys}
    )
    new_dict["iteration"] = torch.LongTensor([iteration])
    print(f"Saved safetensors to {checkpoint_path}")
    save_file(new_dict, checkpoint_path)


def load_safetensors(checkpoint_path, model, for_infer=False):
    """
    Load safetensors model.
    """

    tensors = {}
    iteration = None
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key == "iteration":
                iteration = f.get_tensor(key).item()
            tensors[key] = f.get_tensor(key)
    if hasattr(model, "module"):
        result = model.module.load_state_dict(tensors, strict=False)
    else:
        result = model.load_state_dict(tensors, strict=False)
    for key in result.missing_keys:
        if key.startswith("enc_q") and for_infer:
            continue
        print(f"Missing key: {key}")
    for key in result.unexpected_keys:
        if key == "iteration":
            continue
        print(f"Unexpected key: {key}")
    if iteration is None:
        print(f"Loaded '{checkpoint_path}'")
    else:
        print(f"Loaded '{checkpoint_path}' (iteration {iteration})")
    return model, iteration


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def is_resuming(dir_path):
    # JP-ExtraバージョンではDURがなくWDがあったり変わるため、Gのみで判断する
    g_list = glob.glob(os.path.join(dir_path, "G_*.pth"))
    # d_list = glob.glob(os.path.join(dir_path, "D_*.pth"))
    # dur_list = glob.glob(os.path.join(dir_path, "DUR_*.pth"))
    return len(g_list) > 0


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    try:
        x = f_list[-1]
    except IndexError:
        raise ValueError(f"No checkpoint found in {dir_path} with regex {regex}")
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/base.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r", encoding="utf-8") as f:
            data = f.read()
        with open(config_save_path, "w", encoding="utf-8") as f:
            f.write(data)
    else:
        with open(config_save_path, "r", vencoding="utf-8") as f:
            data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def clean_checkpoints(path_to_models="logs/44k/", n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    import re

    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x):
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (
            x_sorted("G_")[:-n_ckpts_to_keep]
            + x_sorted("D_")[:-n_ckpts_to_keep]
            + x_sorted("WD_")[:-n_ckpts_to_keep]
            + x_sorted("DUR_")[:-n_ckpts_to_keep]
        )
    ]

    def del_info(fn):
        return print(f"Free up space by deleting ckpt {fn}")

    def del_routine(x):
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    # print("config_path: ", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        print(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            print(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def load_model(model_path, config_path):
    hps = get_hparams_from_file(config_path)
    net = SynthesizerTrn(
        # len(symbols),
        108,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to("cpu")
    _ = net.eval()
    _ = load_checkpoint(model_path, net, None, skip_optimizer=True)
    return net


def mix_model(
    network1, network2, output_path, voice_ratio=(0.5, 0.5), tone_ratio=(0.5, 0.5)
):
    if hasattr(network1, "module"):
        state_dict1 = network1.module.state_dict()
        state_dict2 = network2.module.state_dict()
    else:
        state_dict1 = network1.state_dict()
        state_dict2 = network2.state_dict()
    for k in state_dict1.keys():
        if k not in state_dict2.keys():
            continue
        if "enc_p" in k:
            state_dict1[k] = (
                state_dict1[k].clone() * tone_ratio[0]
                + state_dict2[k].clone() * tone_ratio[1]
            )
        else:
            state_dict1[k] = (
                state_dict1[k].clone() * voice_ratio[0]
                + state_dict2[k].clone() * voice_ratio[1]
            )
    for k in state_dict2.keys():
        if k not in state_dict1.keys():
            state_dict1[k] = state_dict2[k].clone()
    torch.save(
        {"model": state_dict1, "iteration": 0, "optimizer": None, "learning_rate": 0},
        output_path,
    )


def get_steps(model_path):
    matches = re.findall(r"\d+", model_path)
    return matches[-1] if matches else None
