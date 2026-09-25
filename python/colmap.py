#!/usr/bin/env python3
"""Create a sparse point cloud from images using pycolmap."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pycolmap


def build_point_cloud(image_dir: Path, output_dir: Path, camera_model: str = "SIMPLE_RADIAL") -> Path:
	"""Run feature extraction, matching, mapping, and export the best model as PLY."""
	image_dir = image_dir.expanduser().resolve()
	output_dir = output_dir.expanduser().resolve()
	database_path = output_dir / "database.db"
	sparse_dir = output_dir / "sparse"
	pointcloud_path = output_dir / "pointcloud.ply"

	if not image_dir.is_dir():
		raise FileNotFoundError(f"Image folder does not exist: {image_dir}")
	images = [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
	if not images:
		raise ValueError(f"No supported images found in {image_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)
	pycolmap.extract_features(
		database_path,
		image_dir,
		camera_mode=pycolmap.CameraMode.AUTO,
		camera_model=camera_model,
	)
	pycolmap.match_exhaustive(database_path)

	reconstructions = pycolmap.incremental_mapping(database_path, image_dir, sparse_dir)
	if not reconstructions:
		raise RuntimeError("COLMAP could not register a reconstruction from the images")

	# Choose the model containing the most registered images.
	reconstruction = max(reconstructions.values(), key=lambda model: model.num_reg_images())
	# Convert COLMAP's sparse reconstruction to an OpenMVS scene, then densify it.
	sparse_model_dir = sparse_dir / "best"
	sparse_model_dir.mkdir(parents=True, exist_ok=True)
	reconstruction.write(sparse_model_dir)
	reconstruction.export_PLY(pointcloud_path)
	if not pointcloud_path.is_file():
		raise RuntimeError(f"Failed to save sparse point cloud to {pointcloud_path}")

	# try:
	# 	import pyopenmvs
	# except ImportError as exc:
	# 	raise RuntimeError("pyopenmvs must be installed to reconstruct and densify the model") from exc

	# mvs_scene = output_dir / "scene.mvs"
	# # Use the OpenMVS binaries distributed in OpenMVS_Ubuntu_x64.
	# openmvs_dir = Path("OpenMVS_Ubuntu_x64").expanduser().resolve()
	# interface_colmap = openmvs_dir / "InterfaceCOLMAP"
	# densify_point_cloud = openmvs_dir / "DensifyPointCloud"
	# if not interface_colmap.is_file() or not densify_point_cloud.is_file():
	# 	raise FileNotFoundError(
	# 		"OpenMVS_Ubuntu_x64 must contain InterfaceCOLMAP and DensifyPointCloud"
	# 	)

	# try:
	# 	subprocess.run(
	# 		[
	# 			str(interface_colmap),
	# 			"-i",
	# 			str(sparse_model_dir),
	# 			"-o",
	# 			str(mvs_scene),
	# 			"--image-folder",
	# 			str(image_dir),
	# 		],
	# 		check=True,
	# 		text=True,
	# 		capture_output=True,
	# 	)
	# except subprocess.CalledProcessError as exc:
	# 	details = (exc.stderr or exc.stdout or "").strip()
	# 	raise RuntimeError(
	# 		f"OpenMVS InterfaceCOLMAP failed with exit code {exc.returncode}"
	# 		f"{': ' + details if details else ''}"
	# 	) from exc

	# try:
	# 	subprocess.run(
	# 		[
	# 			str(densify_point_cloud),
	# 			str(mvs_scene),
	# 			"-o",
	# 			str(output_dir / "scene_dense.mvs"),
	# 		],
	# 		check=True,
	# 		text=True,
	# 		capture_output=True,
	# 	)
	# except subprocess.CalledProcessError as exc:
	# 	details = (exc.stderr or exc.stdout or "").strip()
	# 	raise RuntimeError(
	# 		f"OpenMVS DensifyPointCloud failed with exit code {exc.returncode}"
	# 		f"{': ' + details if details else ''}"
	# 	) from exc

	# dense_pointcloud_path = output_dir / "scene_dense.ply"
	# if not dense_pointcloud_path.exists():
	# 	raise RuntimeError(f"OpenMVS did not produce {dense_pointcloud_path}")

	return pointcloud_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate a sparse point cloud with pycolmap")
	parser.add_argument("images", type=Path, help="Folder containing input images")
	parser.add_argument("output", type=Path, help="Output folder")
	parser.add_argument("--camera-model", default="SIMPLE_RADIAL", help="COLMAP camera model")
	args = parser.parse_args()

	path = build_point_cloud(args.images, args.output, args.camera_model)
	print(f"Point cloud saved to: {path}")


if __name__ == "__main__":
	main()
