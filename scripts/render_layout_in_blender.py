import os
import sys
import json
import argparse


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render a LayoutVLM layout.json in Blender")
    parser.add_argument("--scene_json_file", required=True, help="Path to the original scene JSON used to generate the layout")
    parser.add_argument("--layout_json_file", required=True, help="Path to the generated layout.json with placements")
    parser.add_argument("--asset_dir", required=True, help="Directory containing processed assets (same one used when running LayoutVLM)")
    parser.add_argument("--save_dir", required=True, help="Directory to save render outputs and optional .blend")
    parser.add_argument("--high_res", action="store_true", help="Render at high resolution")
    parser.add_argument("--save_blend", action="store_true", help="Save a .blend file of the scene")
    parser.add_argument("--add_hdri", action="store_true", help="Add HDRI lighting if HDRI file is available")
    parser.add_argument("--side_phi", type=float, default=45.0, help="Side view camera elevation angle in degrees")
    parser.add_argument("--side_indices", type=str, default="3", help="Comma-separated side view indices in [0,1,2,3]")
    # Blender launches scripts as: blender -P script.py -- [args]
    # Respect arguments only after the double dash. If not running under Blender,
    # skip the script name (argv[0]) like argparse normally does.
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    return parser.parse_args(argv)


def main(argv):
    # Ensure repo root on sys.path so we can import project utilities from Blender Python
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    from main import prepare_task_assets  # noqa: E402
    from utils.blender_render import render_existing_scene  # noqa: E402

    args = parse_args(argv)

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.scene_json_file, "r") as f:
        scene_config = json.load(f)

    # Prepare task to include asset paths, metadata, and normalized keys
    task = prepare_task_assets(scene_config, args.asset_dir)

    with open(args.layout_json_file, "r") as f:
        placed_assets = json.load(f)

    # Configure side views
    try:
        side_indices = [int(x) for x in args.side_indices.split(",") if x.strip() != ""]
    except Exception:
        side_indices = [3]

    # Output filenames
    topdown_path = os.path.join(args.save_dir, "top_down_rendering.png")
    sideview_path = None  # Let the util name them per index

    # Run render
    output_images, _ = render_existing_scene(
        placed_assets=placed_assets,
        task=task,
        save_dir=args.save_dir,
        add_hdri=args.add_hdri,
        topdown_save_file=topdown_path,
        sideview_save_file=sideview_path,
        add_coordinate_mark=True,
        annotate_object=False,
        annotate_wall=False,
        render_top_down=True,
        adjust_top_down_angle=None,
        high_res=args.high_res,
        rotate_90=True,
        apply_3dfront_texture=False,
        recenter_mesh=True,
        fov_multiplier=1.3,
        default_font_size=None,
        combine_obj_components=True,
        side_view_phi=args.side_phi,
        side_view_indices=side_indices,
        save_blend=args.save_blend,
        add_object_bbox=False,
        ignore_asset_instance_idx=False,
        floor_material="Travertine008",
    )

    # Summary file
    summary = {
        "top_down": topdown_path,
        "side_views": [p for p in output_images if os.path.basename(p).startswith("side_rendering_")],
        "blend_file": os.path.join(args.save_dir, "scene.blend") if args.save_blend else None,
    }
    with open(os.path.join(args.save_dir, "render_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main(sys.argv)


