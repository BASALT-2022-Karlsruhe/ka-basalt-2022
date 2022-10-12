from genericpath import exists
import json
from pathlib import Path
from tqdm import tqdm
import argparse


def has_early_esc(filepath, n_actions_allowed=200):
    with open(filepath) as demo:
        action_list = list(demo)
    n_actions_after_esc = 0
    esc_pressed = False
    for json_str in action_list:
        try:
            keys = json.loads(json_str)["keyboard"]["keys"]
        except:
            continue
        if "key.keyboard.escape" in keys:
            esc_pressed = True
        elif esc_pressed:
            n_actions_after_esc += 1
        if n_actions_after_esc > n_actions_allowed:
            return True
    return False


def ends_prematurely(filepath, min_actions=100):
    with open(filepath) as demo:
        action_list = list(demo)
    if len(action_list) < min_actions:
        return True


def remove_early_esc(in_path, n_actions_allowed=200, out_path=""):
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

        if "key.keyboard.escape" in keys and n_actions_after_esc > n_actions_allowed:
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


def add_esc(in_path, out_path=""):
    with open(in_path) as demo:
        action_list = list(demo)
    try:
        last_action = json.loads(action_list[-1])
    except:
        pass
    last_keys = last_action["keyboard"]["keys"]
    last_new_keys = last_action["keyboard"]["keys"]
    if not "key.keyboard.escape" in last_keys:
        last_keys.append("key.keyboard.escape")
        last_new_keys.append("key.keyboard.escape")
    action_list[-1] = str(json.dumps(last_action))

    out_path_file = out_path / in_path.parts[-2] / in_path.name if out_path else in_path
    out_path_file.parents[0].mkdir(parents=True, exist_ok=True)
    with open(out_path_file, "w+") as demo:
        demo.writelines(action_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Script for preparing and filtering demonstrations."
    )
    parser.add_argument(
        "Dir",
        metavar="dir",
        type=str,
        help="The directory where demonstrations are saved.",
    )
    parser.add_argument(
        "--remove_early_esc",
        action="store_false",
        help="Whether to remove escape presses that are followed by a certain number of non-escape actions.",
    )
    parser.add_argument(
        "--add_final_esc",
        action="store_false",
        help="Whether to add escape to the final action of demos",
    )
    parser.add_argument(
        "--list_short_demos",
        action="store_false",
        help="Whether to list demos containing less than 'min_actions' actions. The filepaths are saved in 'short_demos.txt'.",
    )
    parser.add_argument(
        "--max_post_esc_actions",
        type=int,
        default=200,
        required=False,
        help="Number of non-escape actions allowed after escape to not remove it.",
    )
    parser.add_argument(
        "--min_actions",
        type=int,
        default=100,
        required=False,
        help="Demos with less than this number of actions will be listed if listing short demos.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        required=False,
        help="Directory where the altered demos will be saved. If not provided, the original demos will be overwritten.",
    )

    args = parser.parse_args()
    dir = args.Dir

    basepath = Path(__file__).parent.resolve()
    if args.output_dir:
        out_path = basepath / args.output_dir
    else:
        out_path = basepath / "prepped_demos"
    out_path.mkdir(parents=True, exist_ok=True)

    traj_paths = get_demo_trajectories(dir)

    if args.remove_early_esc:
        pbar = tqdm(traj_paths)
        pbar.set_description("Removing premature escape presses from demos")
        for path in pbar:
            remove_early_esc(
                path, n_actions_allowed=args.max_post_esc_actions, out_path=out_path
            )

    if args.add_final_esc:
        pbar = tqdm(traj_paths)
        pbar.set_description("Adding escape presses to final action of demos")
        for path in pbar:
            add_esc(path, out_path=out_path)

    short_demo_list = basepath / "short_demos.txt"

    if args.list_short_demos:
        pbar = tqdm(traj_paths)
        pbar.set_description("Listing demos")
        with open(short_demo_list, "w") as f:
            for path in pbar:
                if ends_prematurely(path, min_actions=args.min_actions):
                    f.write(str(path) + "\n")
