import os
import sys
import argparse
import subprocess
import shutil


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Wrapper: run Blender to render a LayoutVLM layout.json")
    parser.add_argument("--scene_json_file", required=True, help="Path to the original scene JSON used to generate the layout")
    parser.add_argument("--layout_json_file", required=True, help="Path to the generated layout.json with placements")
    parser.add_argument("--asset_dir", required=True, help="Directory containing processed assets (same one used when running LayoutVLM)")
    parser.add_argument("--save_dir", required=True, help="Directory to save render outputs and optional .blend")
    parser.add_argument("--high_res", action="store_true", help="Render at high resolution")
    parser.add_argument("--save_blend", action="store_true", help="Save a .blend file of the scene")
    parser.add_argument("--add_hdri", action="store_true", help="Add HDRI lighting if HDRI file is available")
    parser.add_argument("--side_phi", type=float, default=45.0, help="Side view camera elevation angle in degrees")
    parser.add_argument("--side_indices", type=str, default="3", help="Comma-separated side view indices in [0,1,2,3]")
    parser.add_argument("--blender_executable", type=str, default="blender", help="Path to Blender executable (default: 'blender' on PATH)")
    return parser.parse_args(argv[1:])


def abspath(path):
    return os.path.abspath(path) if path is not None else path


def main(argv):
    args = parse_args(argv)

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    blender_script = os.path.join(repo_root, "scripts", "render_layout_in_blender.py")

    scene_json = abspath(args.scene_json_file)
    layout_json = abspath(args.layout_json_file)
    asset_dir = abspath(args.asset_dir)
    save_dir = abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Verify blender executable
    blender_exec = args.blender_executable
    if os.path.sep not in blender_exec:
        # not a path; try which
        if shutil.which(blender_exec) is None:
            raise FileNotFoundError("Blender executable not found on PATH. Set --blender_executable to the full path.")
    else:
        if not os.path.exists(blender_exec):
            raise FileNotFoundError(f"Blender executable not found: {blender_exec}")

    # Build command
    cmd = [
        blender_exec,
        "-b",
        "-P", blender_script,
        "--",
        "--scene_json_file", scene_json,
        "--layout_json_file", layout_json,
        "--asset_dir", asset_dir,
        "--save_dir", save_dir,
        "--side_phi", str(args.side_phi),
        "--side_indices", str(args.side_indices),
    ]
    if args.high_res:
        cmd.append("--high_res")
    if args.save_blend:
        cmd.append("--save_blend")
    if args.add_hdri:
        cmd.append("--add_hdri")

    # Run Blender
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main(sys.argv)



