from laminated_dust3r.model import LaminatedDust3rModel
from dust3r.demo import get_reconstructed_scene
from laminated_dust3r.reconstruction import get_reconstructed_scene_laminated

import click
import functools
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import cv2


def load_config(config_path):
    if config_path is None:
        # Default config file path
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        print(f"Config file not found at {config_path}. Using base config.")
    return {}


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(),
    required=False,
    help="Config file",
    default=None,
)
@click.option(
    "--name",
    "-n",
    type=click.STRING,
    required=False,
    help="Name of the run",
    default=None,
)
@click.option(
    "--export_path",
    "-e",
    type=click.Path(),
    required=False,
    help="Export path",
    default=None,
)
@click.option(
    "--device",
    "-d",
    type=click.STRING,
    required=False,
    help="Device to use",
    default=None,
)
@click.option(
    "--image_folder",
    "-d",
    type=click.Path(),
    required=False,
    help="Image folder",
    default=None,
)
@click.option(
    "--exclude_imgs",
    "-e",
    type=click.STRING,
    required=False,
    help="Images to exclude",
    default=None,
)
@click.option(
    "--primary_image",
    "-p",
    type=click.STRING,
    required=False,
    help="Primary image",
    default=None,
)
@click.option(
    "--image_order",
    "-o",
    type=click.STRING,
    required=False,
    help="Image order",
    default=None,
)
def main(
    config,
    name,
    export_path,
    device,
    image_folder,
    exclude_imgs,
    primary_image,
    image_order,
):
    config = load_config(config)
    name = name or config.get("name", "base")
    export_path = export_path or config.get("export_path", "./output")
    device = device or config.get("device", "cpu")
    image_folder = image_folder or config.get("image_folder", "./data/")
    exclude_imgs = exclude_imgs or config.get("exclude_imgs", None)
    primary_image = primary_image or config.get("primary_image", None)
    image_order = image_order or config.get("image_order", None)

    # make directory for meshes
    os.makedirs(export_path, exist_ok=True)

    # get list of images
    image_list = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f))
    ]

    # exclude on the basis of exclude_imgs
    if exclude_imgs is not None:
        exclude_imgs = exclude_imgs.split(",")
        exclude_imgs = [img.replace(" ", "") for img in exclude_imgs]
        image_list = [
            img
            for img in image_list
            if img.split("/")[-1].replace(" ", "") not in exclude_imgs
        ]

    # set primary image
    assert not (
        primary_image and image_order
    ), "Only one of primary_image and image_order can be set"
    if primary_image is not None:
        primary_image = primary_image.replace(" ", "")
        image_list_names = [img.split("/")[-1].replace(" ", "") for img in image_list]
        if primary_image not in image_list_names:
            raise ValueError("Primary image not in image list")
        primary_image_idx = image_list_names.index(primary_image)
        image_list = [image_list[primary_image_idx]] + [
            image_list[i] for i in range(len(image_list)) if i != primary_image_idx
        ]
        print("Primary image set to: ", image_list[0])
    elif image_order is not None:
        image_order = image_order.split(",")
        image_order = [img.replace(" ", "") for img in image_order]
        image_list_names = [img.split("/")[-1].replace(" ", "") for img in image_list]
        # check image order is a subset of image_list_names
        assert set(image_order).issubset(
            set(image_list_names)
        ), f"Image order contains invalid names: {set(image_order) - set(image_list_names)}"
        image_list = [image_list[image_list_names.index(img)] for img in image_order]
        print("Image order set to: ", image_list)
    else:
        print("No image order given, using first image as primary: ", image_list[0])

    num_images = len(image_list)

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

    out_data = recon_fun(**reconstruction_params)

    # now we want to export data
    for i in range(num_images):
        key = "pts3d" if i == 0 else "pts3d_in_other_view"
        depth = out_data[f"pred{i+1}"][key][0, :, :, -1].detach().cpu().numpy()
        conf = out_data[f"pred{i+1}"]["conf"].detach().cpu().numpy().squeeze()
        mx_depth = np.percentile(depth, 90)
        mn_depth = np.percentile(depth, 10)
        depth = (depth - mn_depth) / (mx_depth - mn_depth)

        gs = GridSpec(2, num_images * 2, height_ratios=[num_images, 2])

        # save depth map
        ax_depth = plt.subplot(gs[0, :num_images])
        ax_depth.imshow(depth, cmap="hsv", vmin=0, vmax=1)
        ax_depth.axis("off")
        ax_depth.set_title(f"depth {i}")

        # save confidence map
        ax_conf = plt.subplot(gs[0, num_images:])
        ax_conf.imshow(conf, cmap="bwr", vmin=0, vmax=25)
        ax_conf.axis("off")
        ax_conf.set_title(f"confidence {i}")

        # save reference images
        for j in range(num_images):
            im = cv2.imread(image_list[j])
            if j == 0:
                cv2.rectangle(
                    im,
                    (0, 0),
                    (im.shape[1], im.shape[0]),
                    (0, 0, 255),
                    10,
                )
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ax_ref = plt.subplot(gs[1, 2 * j : 2 * j + 2])
            ax_ref.imshow(im)
            ax_ref.axis("off")

        plt.suptitle(name)
        plt.tight_layout()
        plt.savefig(os.path.join(export_path, f"depth_map_{i}.png"))

        plt.clf()
        plt.close()

        # save point cloud
        pts3d = out_data[f"pred{i+1}"][key][0].detach().cpu().numpy()

        # save as .obj file
        with open(os.path.join(export_path, f"point_cloud_{i}.obj"), "w") as f:
            for pt in pts3d.reshape(-1, 3):
                f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
            for i in range(pts3d.shape[0] - 1):
                for j in range(pts3d.shape[1] - 1):
                    if conf[i, j] > 3 or True:
                        f.write(
                            f"f {i*pts3d.shape[1]+j+1} {i*pts3d.shape[1]+j+2} {(i+1)*pts3d.shape[1]+j+2}\n"
                        )
                        f.write(
                            f"f {i*pts3d.shape[1]+j+1} {(i+1)*pts3d.shape[1]+j+2} {(i+1)*pts3d.shape[1]+j+1}\n"
                        )


if __name__ == "__main__":
    main()
