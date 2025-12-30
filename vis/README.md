# 6D pose visualization (OpenGL implementation)

Install the conda environment for visualization

```
conda env create --name vis --file=environment.yaml
```
## Render the pose
```
python renderPose_v3.py --debug ... --md_store_path ... --op ... --res ... --srv_res_path ...  --vis_inliers_2D ... --vis_inliers_3D ...
```

where:
- [no]debug: Run in debug mode
    (default: 'false')
- md_store_path: Path the the ply models are stored
    (default: '')
- op: Controls the opacity of the overlayed pose
    (default: '0.7')
    (a number)
- res: Resolution of input/output image
    (default: '640,480')
    (a comma separated list)
- srv_res_path: Service results path, usually the folder /root/sanbox/results/XXXXX/image_raw
    (default: '')
- [no]vis_inliers_2D: Visualize the inliers in 2D
    (default: 'true')
- [no]vis_inliers_3D: Visualize the inliers in 3D
    (default: 'false')

Minimum required are srv_res_path, md_store_path. For visualization, please rename the object models as such, e.g. obj_000001.ply should be renamed to 1.ply. 


## Visualizing the inliers

```bash
python inliers_vis.py --corPath ... --imgPath ... --resPath ...
```

where:

- **corPath** is the path where the correspodence file is stored (e.g Test_1/Obj_1/corr_)
- **imgPath** the path of the raw image (e.g Test_1/Obj_1/image_raw.png)
- **resPath** output path to save the result image (e.g Test_1/Obj_1/inliers.png)

## Example use case

### Pose visualization

```
python3 renderPose_v3.py --srv_res_path /path/to/example/image_raw/ --md_store_path /path/to/models
```

The script renders the pose for every subfolder Test_XXXX_YY_ZZ_IIIII in `srv_res_path`. Suppose `srv_res_path`is `./example/image_raw/` and contains the subfolder `Test_2023_09_20_100126/Obj_1/`. Then the result is an image called `result.png` inside `./example/image_raw/Test_2023_09_20_100126/Obj_1/`

If we had more test folders then each result image would be created in 
`./example/image_raw/Test_XXXX_YY_ZZ_IIIII/Obj_<ID>/`.

### Inliers visualization

```bash
python inliers_vis.py --corPath './example/image_raw/Test_2023_09_20_100126/Obj_1/corr_' --imgPath './example/image_raw/Test_2023_09_20_100126/Obj_1/image_raw.png' --resPath './example/image_raw/Test_2023_09_20_100126/Obj_1/inliers.png'
```

The result is an image called inliers.png inside `/example/image_raw/Test_2023_09_20_100126/Obj_1`.
