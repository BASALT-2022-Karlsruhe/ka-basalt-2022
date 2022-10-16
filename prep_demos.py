import json
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2


def is_short(filepath, min_actions=100):
    with open(filepath) as demo:
        action_list = list(demo)
    if len(action_list) < min_actions:
        return True


def get_length_diff(traj):
    with open(traj) as demo:
        action_list = list(demo)
    len_traj = len(action_list)

    vid = traj.with_suffix(".mp4")
    cap = cv2.VideoCapture(str(vid))
    len_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return len_traj - len_vid


def remove_early_esc(in_path, out_path, percent_actions_allowed=10):
    with open(in_path) as demo:
        action_list = list(demo)
    n_actions_after_esc = 0
    for idx in range(-1, -len(action_list) - 1, -1):
        try:
            action = json.loads(action_list[idx])
        except:
            continue
        keys = action["keyboard"]["keys"]
        new_keys = action["keyboard"]["newKeys"]

        if (
            "key.keyboard.escape" in keys
            and n_actions_after_esc > percent_actions_allowed
        ):
            keys.remove("key.keyboard.escape")
            if "key.keyboard.escape" in new_keys:
                new_keys.remove("key.keyboard.escape")
            action_list[idx] = str(json.dumps(action))
        else:
            n_actions_after_esc += 1
    out_path_file = out_path / in_path.parts[-2] / in_path.name if out_path else in_path
    out_path_file.parents[0].mkdir(parents=True, exist_ok=True)
    with open(out_path_file, "w+") as demo:
        demo.writelines(action_list)


def get_demo_trajectories(dir):
    basepath = Path(dir).absolute().resolve()
    trajectories = []
    for path in basepath.rglob("*"):
        if path.suffix == ".jsonl":
            trajectories.append(path)
    return trajectories


def is_unfinished(traj, percent_actions_allowed=10):
    with open(traj) as demo:
        action_list = list(demo)
    n_actions_allowed = int(len(action_list) * percent_actions_allowed / 100) + 1
    for action in action_list[:-n_actions_allowed:-1]:
        try:
            action = json.loads(action)
        except:
            continue
        if "key.keyboard.escape" in action["keyboard"]["keys"]:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script for preparing and filtering demonstrations. Filtered demos are saved in 'bad_demos.txt'. Altered demos are saved in 'prepped_demos.txt'."
    )
    parser.add_argument(
        "Dir",
        metavar="dir",
        type=str,
        help="The directory where demonstrations are saved.",
    )
    parser.add_argument(
        "--remove_early_esc",
        action="store_true",
        help="Add flag to remove escape presses that are followed by a certain number of non-escape actions.",
    )
    parser.add_argument(
        "--list_short",
        action="store_false",
        help="Add flag to not list demos containing less than 'min_actions' actions.",
    )
    parser.add_argument(
        "--list_unfinished",
        action="store_false",
        help="Add flag to not list demos that don't have an esc within the final '--percent_actions_allowed'.",
    )
    parser.add_argument(
        "--percent_actions_allowed",
        type=int,
        default=10,
        required=False,
        help="Percentage of non-escape actions relative to number of total actions allowed after escape to not remove it.",
    )
    parser.add_argument(
        "--min_actions",
        type=int,
        default=100,
        required=False,
        help="Demos with less than this number of actions will be listed if listing short demos.",
    )

    args = parser.parse_args()
    dir = args.Dir

    basepath = Path(__file__).parent.resolve()

    out_path = basepath / "prepped_demos"
    out_path.mkdir(parents=True, exist_ok=True)
    traj_paths = get_demo_trajectories(dir)

    bad_demo_list = basepath / "bad_demos.txt"

    if args.list_unfinished or args.list_short:
        pbar = tqdm(traj_paths)
        pbar.set_description("Listing bad demos")
        with open(bad_demo_list, "w") as f:
            for path in pbar:
                list_demo = (is_unfinished(path) and args.list_unfinished) or (
                    is_short(path) and args.list_short
                )
                if list_demo:
                    f.write(str(path) + "\n")

    if args.remove_early_esc:
        pbar = tqdm(traj_paths)
        pbar.set_description("Removing premature escape presses from demos")
        for path in pbar:
            remove_early_esc(
                path,
                out_path=out_path,
                percent_actions_allowed=args.percent_actions_allowed,
            )
