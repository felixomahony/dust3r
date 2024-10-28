from dust3r.demo import *


def get_reconstructed_scene_laminated(
    outdir,
    model,
    device,
    silent,
    image_size,
    filelist,
    schedule,
    niter,
    min_conf_thr,
    as_pointcloud,
    mask_sky,
    clean_depth,
    transparent_cams,
    cam_size,
    scenegraph_type,
    winsize,
    refid,
):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        raise ValueError("Need at least 2 images to reconstruct a scene")
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_ns(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    # pairs.append(pairs[-1][::-1])
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    return output

    # mode = (
    #     GlobalAlignerMode.PointCloudOptimizer
    #     if len(imgs) > 2
    #     else GlobalAlignerMode.PairViewer
    # )
    # scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    # lr = 0.01

    # if mode == GlobalAlignerMode.PointCloudOptimizer:
    #     loss = scene.compute_global_alignment(
    #         init="mst", niter=niter, schedule=schedule, lr=lr
    #     )

    # outfile = get_3D_model_from_scene(
    #     outdir,
    #     silent,
    #     scene,
    #     min_conf_thr,
    #     as_pointcloud,
    #     mask_sky,
    #     clean_depth,
    #     transparent_cams,
    #     cam_size,
    # )

    # # also return rgb, depth and confidence imgs
    # # depth is normalized with the max value for all images
    # # we apply the jet colormap on the confidence maps
    # rgbimg = scene.imgs
    # depths = to_numpy(scene.get_depthmaps())
    # confs = to_numpy([c for c in scene.im_conf])
    # cmap = pl.get_cmap("jet")
    # depths_max = max([d.max() for d in depths])
    # depths = [d / depths_max for d in depths]
    # confs_max = max([d.max() for d in confs])
    # confs = [cmap(d / confs_max) for d in confs]

    # imgs = []
    # for i in range(len(rgbimg)):
    #     imgs.append(rgbimg[i])
    #     imgs.append(rgb(depths[i]))
    #     imgs.append(rgb(confs[i]))

    # return scene, outfile, imgs
