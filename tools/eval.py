import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
from munch import munchify
from rich.console import Console
from rich.progress import track
from rich.table import Table

sys.path.append(str(Path(__file__).resolve().parent.parent))

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.deform_model import DeformModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.camera_utils import Camera, getProjectionMatrix2
from utils.dataset import load_dataset
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

console = Console()
device = "cuda"


def evaluate(path_str, dataset_path=None):
    path = Path(path_str)
    console.rule(f"[bold green]Evaluating results from: {path}")

    config = yaml.safe_load((path / "config.yml").open())
    is_2dgs = config["Training"]["4dtam"]["is_2dgs"]
    model_params = munchify(config["model_params"])
    pipeline_params = munchify(config["pipeline_params"])

    gaussians = GaussianModel(model_params.sh_degree, config=config, is_2dgs=is_2dgs)
    gaussians.load_ply(str(path / "point_cloud/final/point_cloud.ply"))

    warpfield = DeformModel(is_2dgs=is_2dgs)
    warpfield.deform.load_state_dict(torch.load(path / "warpfield_after.pth"))
    
    if dataset_path is not None:
        print("Overriding dataset path with:", dataset_path)
        base_path = Path(dataset_path)
        # config["Dataset"]["dataset_path"] = base_path
    else:
        raise Exception("Please provide --dataset to eval.py")
        # print("Dataset path from config:", config["Dataset"]["dataset_path"])
        # # get base path from config (remove 'train' if present)
        # base_path = config["Dataset"]["dataset_path"]
        # if "train" in base_path:
        #     base_path = Path(base_path).parent
        #     # config["Dataset"]["dataset_path"] = base_path
    
    print("Dataset path:", base_path)
    
    config_train = config.copy()
    config_train["Dataset"]["dataset_path"] = base_path / "train"
    dataset_train = load_dataset(
        model_params,
        str(config_train["Dataset"]["dataset_path"]),
        config=config_train)
    print("len(dataset_train)", len(dataset_train))
    
    # dataset = load_dataset(model_params, model_params.source_path, config=config)
    projection_matrix = (
        getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=dataset_train.fx,
            fy=dataset_train.fy,
            cx=dataset_train.cx,
            cy=dataset_train.cy,
            W=dataset_train.width,
            H=dataset_train.height,
        )
        .transpose(0, 1)
        .to(device=device)
    )
    
    config_test = config.copy()
    config_test["Dataset"]["dataset_path"] = base_path / "test"
    if (base_path / "test").exists():
        print("Found test set:", base_path / "test")
        dataset_test = load_dataset(
            model_params,
            str(config_test["Dataset"]["dataset_path"]),
            config=config_test)
        print("len(dataset_test)", len(dataset_test))
    else:
        console.print(f"[bold yellow]Warning:[/bold yellow] No test set found at {base_path / 'test'}. Exiting evaluation.")
        dataset_test = None
        
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)

    def eval_dataset(dataset):
        
        renders = []
        psnr_scores, ssim_scores, lpips_scores, depth_l1_scores = [], [], [], []

        for idx in track(range(len(dataset)), description="Evaluating test views..."):
            
            print("Frame:", idx)

            cam = Camera.init_from_dataset(dataset, idx, projection_matrix)
            cam.T = cam.T_gt
            fid = torch.tensor([cam.fid], device=device)
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            d_xyz, d_rot, d_scale = warpfield.step(gaussians.get_xyz.detach(), time_input)
            render_result = render(
                cam,
                gaussians,
                pipeline_params,
                bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device=device),
                d_xyz=d_xyz,
                d_rotation=d_rot,
                d_scaling=d_scale,
                is_2dgs=is_2dgs,
            )

            gt_img = cam.original_image
            pred_img = torch.clamp(render_result["render"], 0.0, 1.0)

            mask = gt_img > 0
            psnr_scores.append(
                psnr(pred_img[mask].unsqueeze(0), gt_img[mask].unsqueeze(0)).item()
            )
            ssim_scores.append(ssim(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).item())
            lpips_scores.append(
                lpips_metric(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).item()
            )

            gt_depth = torch.from_numpy(cam.depth).float().to(device)[None]
            pred_depth = render_result["depth"]
            mask_depth = gt_depth > 0
            depth_l1 = l1_loss(
                pred_depth[mask_depth].unsqueeze(0), gt_depth[mask_depth].unsqueeze(0)
            ).item()
            depth_l1_scores.append(depth_l1)
            
            # put gt and pred side by side and save to disk
            gt_img_np = (gt_img.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            pred_img_np = (pred_img.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            combined = np.concatenate((gt_img_np, pred_img_np), axis=1)
            renders.append(combined)

            torch.cuda.empty_cache()

        result = {
            "mean_psnr": float(np.mean(psnr_scores)),
            "mean_ssim": float(np.mean(ssim_scores)),
            "mean_lpips": float(np.mean(lpips_scores)),
            "mean_depth_l1": float(np.mean(depth_l1_scores))
        }
        
        return result, renders

    result_train, renders_train = eval_dataset(dataset_train)
    console.print("\n[bold green]✅ Training set evaluation complete!")
    split_results = {"train": result_train}
    split_renders = {"train": renders_train}
    
    if dataset_test is not None:
        result_test, renders_test = eval_dataset(dataset_test)
        console.print("\n[bold green]✅ Test set evaluation complete!")
        split_results["test"] = result_test
        split_renders["test"] = renders_test
    
    # ate_stats_path = path / "plot" / "stats_final.json"
    # if not ate_stats_path.exists():
    #     console.print(f"[bold yellow]Warning:[/bold yellow] {ate_stats_path} does not exist.")
    #     # return
    # else:
    #     console.print(f"[bold green]Found ATE stats at:[/bold green] {ate_stats_path}")
    #     ate_stats = json.load(ate_stats_path.open())
    #     result["rmse"] = ate_stats["rmse"]
    
    for split_name, result in split_results.items():

        json_path = path / f"{split_name}_final_result.json"
        summary_path = path / f"{split_name}_summary.json"
        vis_dir = path / f"{split_name}_renders"

        mkdir_p(path)
 
        # save renders to vis_dir
        if split_name in split_renders:
            mkdir_p(vis_dir)
            for i, img in enumerate(split_renders[split_name]):
                img_path = vis_dir / f"{i:04d}.png"
                # save using cv2
                import cv2
                # rgb to bgr
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(img_path), img)
            console.print(f"[bold green]✅ Saved renders to {vis_dir}")
 
        json.dump(result, json_path.open("w", encoding="utf-8"), indent=4)
        json.dump({str(path): result}, summary_path.open("w", encoding="utf-8"), indent=4)

        table = Table(title="Evaluation Results", show_lines=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bold white", justify="right")
        for key, value in result.items():
            table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
        console.print(table)

    console.print("\n[bold green]✅ Results saved to:")
    console.print(f"- {json_path}")
    console.print(f"- {summary_path}")

    console.print(
        f"\n[bold green]✅ Evaluation complete! Results saved to:\n- {json_path}\n- {summary_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 4DTAM result")
    parser.add_argument("--path", type=str, required=True, help="Path to result folder")
    parser.add_argument("--dataset", type=str, required=False, help="Path to dataset folder (overrides config)")
    args = parser.parse_args()

    evaluate(args.path, dataset_path=args.dataset)
