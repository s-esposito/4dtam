import argparse
import json
import time
from pathlib import Path
import sys
import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify
from rich.console import Console
from rich.panel import Panel

console = Console()

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from gui import gui_utils, slam_gui
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.deform_model import DeformModel
from utils.camera_utils import CameraMsg

DEVICE = "cuda"


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_trajectory(trj_file):
    with open(trj_file, "r") as f:
        data = json.load(f)
    trj_est = torch.tensor(data["trj_est"], device=DEVICE)
    trj_gt = torch.tensor(data["trj_gt"], device=DEVICE)
    return data, trj_est, trj_gt


def create_keyframes(trj_json, trj_est, trj_gt):
    trj_len = len(trj_json["trj_id"])
    total_frames = trj_json["trj_id"][-1] + 1
    return [
        CameraMsg(
            Camera=None,
            T=torch.linalg.inv(trj_est[i]),
            T_gt=torch.linalg.inv(trj_gt[i]),
            uid=i,
            fid=i / total_frames,
        )
        for i in range(trj_len)
    ]


def visualize(path_str):
    path = Path(path_str)
    console.rule(f"[bold green]Loading result from: {path}")

    config = load_config(path / "config.yml")
    trj_json, trj_est, trj_gt = load_trajectory(path / "plot" / "trj_final.json")

    is_2dgs = config["Training"]["4dtam"]["is_2dgs"]
    model_params = munchify(config["model_params"])
    pipeline_params = munchify(config["pipeline_params"])
    seq_name = config["Dataset"]["dataset_path"].split("/")[-2]

    with torch.no_grad():
        gaussians = GaussianModel(
            model_params.sh_degree, config=config, is_2dgs=is_2dgs
        )
        gaussians.load_ply(str(path / "point_cloud/final/point_cloud.ply"))
        console.log("[bold cyan]Loaded Gaussian Splats.")

    warpfield = DeformModel()
    warpfield.deform.load_state_dict(torch.load(path / "warpfield_after.pth"))
    console.log("[bold cyan]Loaded warpfield weights.")

    q_main2vis = mp.Queue()
    q_vis2main = mp.Queue()
    params_gui = gui_utils.ParamsGUI(
        pipe=pipeline_params,
        background=torch.tensor([0, 0, 0], dtype=torch.float32, device=DEVICE),
        gaussians=gaussians,
        q_main2vis=q_main2vis,
        q_vis2main=q_vis2main,
        is_2dgs=is_2dgs,
        seq_name=seq_name,
    )

    gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
    gui_process.start()
    console.log("[green]Started GUI process.")

    keyframes = create_keyframes(trj_json, trj_est, trj_gt)

    q_main2vis.put(
        gui_utils.GaussianPacket(
            current_frame=keyframes[0], warpfield=warpfield.deform.state_dict()
        )
    )
    time.sleep(1)
    q_main2vis.put(
        gui_utils.GaussianPacket(
            keyframes=keyframes, warpfield=warpfield.deform.state_dict()
        )
    )
    console.log("[green]Sent Gaussian packets to GUI.")

    console.print(
        Panel.fit(
            "✅ [bold green]Visualization running. Waiting for GUI process to end..."
        )
    )
    gui_process.join()
    console.log("[bold green]GUI process finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visualize 4DTAM results")
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the result directory"
    )
    args = parser.parse_args()

    visualize(args.path)
