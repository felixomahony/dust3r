from laminated_dust3r.model import LaminatedDust3rModel
from dust3r.demo import get_reconstructed_scene
from laminated_dust3r.reconstruction import get_reconstructed_scene_laminated

import click
import functools
import os


@click.command()
@click.option(
    "--export_path",
    "-e",
    type=click.Path(),
    required=True,
    help="Export path",
    default="./meshes",
)
@click.option(
    "--device",
    "-d",
    type=click.STRING,
    required=False,
    help="Device to use",
    default="cuda",
)
@click.option(
    "--image_folder",
    "-d",
    type=click.Path(),
    required=True,
    help="Image folder",
    default="./input_images",
)
def main(export_path, device, image_folder):
    # make directory for meshes
    os.makedirs(export_path, exist_ok=True)

    # get list of images
    image_list = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f))
    ]

    # set some default values
    scenegraph_type = "complete"
    winsize = 1
    refid = 0

    reconstruction_params = {
        "filelist": image_list,
        "schedule": "linear",
        "niter": 300,
        "min_conf_thr": 3,
        "as_pointcloud": False,
        "mask_sky": False,
        "clean_depth": True,
        "transparent_cams": True,
        "cam_size": 0.05,
        "scenegraph_type": scenegraph_type,
        "winsize": winsize,
        "refid": refid,
    }

    model = LaminatedDust3rModel.default(device=device)
    recon_fun = functools.partial(
        get_reconstructed_scene_laminated, export_path, model, device, False, 512
    )
    out_model = recon_fun(**reconstruction_params)


if __name__ == "__main__":
    main()
