# Script for segmentation of demos
import cv2
import os
from tqdm import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt

# splits and saves segments based on keyframes
def saveFiles(path, output_dir, filename, fps, frames, keyframes):
    if not frames:
        print("Empty frame - return.")
        return

    # check for and create dirs
    for i in range(len(keyframes) + 1):
        p = f"{output_dir}/stage_{i+1}"
        if not os.path.exists(p):
            os.makedirs(p)

    # write files
    keyframes = [0, *keyframes, -1]
    for i, f in enumerate(keyframes):
        if f == 0:
            continue

        # write video

        try:
            (frames[0].shape[1], frames[0].shape[0])
        except:
            print("error: frames", frames)

        video_out = cv2.VideoWriter(
            f"{output_dir}/stage_{i}/{filename}",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (frames[0].shape[1], frames[0].shape[0]),
            1,
        )

        for frame in frames[keyframes[i - 1] : keyframes[i]]:
            video_out.write(frame)

        # write actions jsonl
        lines = open(f"{path[:-4]}.jsonl", "r").readlines()
        with open(f"{output_dir}/stage_{i}/{filename[:-4]}.jsonl", "w") as file:
            for line in lines[keyframes[i - 1] : keyframes[i]]:
                file.write(line)


# Split MakeWaterfall demos into 2 stages based on when the bucket is emptied
def SplitWaterfallDemo(path, output_dir, out_name):

    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # get FPS
    fps = vidObj.get(cv2.CAP_PROP_FPS)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    # keyframe for splitting
    keyframe = -1

    # "framebuffer"
    frames = []

    # check inventory frame by frame
    while success:
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if not success:
            break

        # mean color of the bucket liquid
        mean_color = image[344:348, 237:243, :].reshape((24, 3)).mean(0)

        # check for maximum color value (could do this differently, but it works)
        if mean_color.max() > 140:
            # theres water in the bucket
            if keyframe != -1:
                # this makes sure we only split at the last emptying of the bucket
                keyframe = -1
        else:
            # bucket is empty
            if keyframe == -1:
                # bucket was full in the previous frame
                keyframe = count

        # store frame
        frames.append(image)
        count += 1

    # abort if demo reaches maximum length
    if len(frames) >= 6000:
        return

    # save files
    saveFiles(path, output_dir, out_name, fps, frames, [keyframe])


# Split FindCave demos into 2 stages based on time. Last 10 seconds are approach, everything before is exploration.
def SplitFindCaveDemo(path, output_dir, out_name, darkness=True, plot=False):

    last_n_seconds = 2
    darkness_mean_threshold = 40
    darkness_var_threshold = 250
    window = 30  # frames

    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # get FPS
    fps = vidObj.get(cv2.CAP_PROP_FPS)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    # keyframe for splitting
    keyframe = -1

    # "framebuffer"
    frames = []
    means = []
    running_mean = 255
    vars = []

    # collect frames
    while success:

        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if not success:
            break

        if darkness and keyframe < 0:
            greyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean, var = greyImage.mean(), greyImage.var()
            means.append(mean)
            vars.append(var)
            running_mean += 1 / window * (mean - running_mean)
            #print(
            #    f"Grey frame {count}: ema {running_mean:.2f}, mean {mean:.2f}, var {var:.2f}",
            )
            if running_mean < darkness_mean_threshold or (
                mean < darkness_mean_threshold and var < darkness_var_threshold
            ):
                keyframe = count

        frames.append(image)
        count += 1

    # abort if demo reaches maximum length
    if len(frames) >= 3600:
        return
    # print(len(frames))
    if not darkness or keyframe < 0:
        keyframe = int(len(frames) - last_n_seconds * fps)

    seconds_from_start = keyframe / fps
    seconds_from_end = len(frames) / fps - seconds_from_start
    frames_from_end = len(frames) - keyframe
    print(
        f"Splitting {path} at frame {keyframe} = {frames_from_end} frames from end = {seconds_from_start:.2f} s from start = {seconds_from_end:.2f} s from end"
    )

    if plot:
        plt.figure()
        bins = np.linspace(0, 100, 100)
        plt.hist(means[:keyframe], bins, alpha=0.5, label="exploration")
        plt.hist(means[keyframe:], bins, alpha=0.5, label="cave")
        plt.xlabel("mean intensity")
        plt.legend()
        plt.figure()
        bins2 = np.linspace(0, 3000, 100)
        plt.hist(vars[:keyframe], bins2, alpha=0.5, label="exploration")
        plt.hist(vars[keyframe:], bins2, alpha=0.5, label="cave")
        plt.legend()
        plt.xlabel("var intensity")
        plt.show()

    saveFiles(path, output_dir, out_name, fps, frames, [keyframe])


# Splits CreateVillageAnimalPen demos into 3 stages. prior to construction (find place and sometimes collect animals), construction, and bring animals to pen
def SplitCreateVillageAnimalPenDemo(path, output_dir, out_name):
    SplitInventoryChangeDemo(
        [path], output_dir, out_name, slots=[0, 1], max_len=6000, pen_building=True
    )


# Splits BuildVillageHouse demos into 3 stages. prior to construction (find place), construction, and house tour
def SplitBuildVillageHouseDemo(path, output_dir, out_name):
    SplitInventoryChangeDemo(
        path, output_dir, out_name, slots=[2, 3, 4, 5, 6, 7, 8], max_len=14400
    )


# Split demos into 3 stages based on inventory changes, slots defines which item slots to watch.
def SplitInventoryChangeDemo(
    paths, output_dir, out_name, slots=[0, 1], max_len=6000, pen_building=False
):

    # Used as counter variable
    count = 0

    keyframe = -1

    # "framebuffer"
    frames = []

    # stores first item config
    first = None

    # stores last/previous item config
    last = None

    # timestamps of inventory changes
    changes = []

    diff_to_first = []
    gate_built = False

    for path in paths:
        # Path to video file
        vidObj = cv2.VideoCapture(path)

        actions = [json.loads(x) for x in open(f"{path[:-4]}.jsonl", "r").readlines()]
        linecount = -1
        # get FPS
        fps = vidObj.get(cv2.CAP_PROP_FPS)

        # checks whether frames were extracted
        success = 1

        craft = False
        while success:

            # vidObj object calls read
            # function extract frames
            success, image = vidObj.read()
            if not success:
                break

            frames.append(image)
            count += 1
            linecount += 1

            if count < 10:
                # skip first 10 frames (only in analysis)
                continue

            # detect and skip crafting menu
            # if np.sum(np.abs(image[110:155,310:327, :].reshape((765,3)).mean(0) - np.array([197, 197, 197]))) < 5 or np.sum(np.abs(image[160:178,370:400,:].reshape((540,3)).mean(0) - np.array([197, 197, 197]))) < 5 or np.sum(np.abs(image[102:120,380:398,:].reshape((324,3)).mean(0) - np.array([197, 197, 197]))) < 5:
            #    craft = True
            #    continue

            if actions[min(linecount, len(actions) - 1)]["isGuiOpen"]:
                craft = True
                continue

            if craft:
                craft = False
                continue

            # switch to black and white as we are only checking for changes in item numbers!
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(
                grayImage, 220, 255, cv2.THRESH_BINARY
            )

            if first is None:
                # save first item config for each slot
                first = []
                for slot in slots:
                    first.append(
                        blackAndWhiteImage[350:357, 237 + 20 * slot : 248 + 20 * slot]
                    )
            else:
                # calculate diff for each slot w.r.t. initial config
                diff = 0
                init_diff = []
                for i, slot in enumerate(slots):
                    slot_diff = np.sum(
                        np.abs(
                            cv2.subtract(
                                first[i],
                                blackAndWhiteImage[
                                    350:357, 237 + 20 * slot : 248 + 20 * slot
                                ],
                            )
                        )
                    )
                    init_diff.append(slot_diff)
                    diff += slot_diff
                # reset in case we are back to initial config (this is mainly to filter out noise)
                if diff == 0 and keyframe != -1:
                    keyframe = -1
                    changes = []

                # check for change over initial config
                if diff > 200:
                    # change detected
                    if keyframe == -1:
                        # log timestamp of change
                        keyframe = count
                        changes.append(count)

                        # save new config
                        last = []
                        for slot in slots:
                            last.append(
                                blackAndWhiteImage[
                                    350:357, 237 + 20 * slot : 248 + 20 * slot
                                ]
                            )

                    # get current config (this is done because the diff calculation cv2.subtract will change the image)
                    current = []
                    for slot in slots:
                        current.append(
                            blackAndWhiteImage[
                                350:357, 237 + 20 * slot : 248 + 20 * slot
                            ]
                        )

                    # calculate diff with previous config
                    diff = 0
                    for i, slot in enumerate(slots):
                        diff += np.sum(
                            np.abs(
                                cv2.subtract(
                                    last[i],
                                    blackAndWhiteImage[
                                        350:357, 237 + 20 * slot : 248 + 20 * slot
                                    ],
                                )
                            )
                        )

                    if diff > 200:
                        if pen_building:
                            key_slot = 1
                        else:
                            key_slot = 3

                        # cv2.imwrite("frames/frame%d.jpg" % count, blackAndWhiteImage[350:357,237+20*slots[key_slot]:248+20*slots[key_slot]])
                        diff_to_first.append(init_diff[key_slot])
                        if pen_building and init_diff[key_slot] != 0:
                            if gate_built:
                                return
                            else:
                                gate_built = True
                        # next change detected
                        # save config
                        last = current
                        # log timestamp for change
                        changes.append(count)

    # abort if demo reaches maximum length or not enough changes were detected for segmentation
    if len(frames) >= max_len or len(changes) < 2:
        return
    # abort if gate is not last build action
    if pen_building and diff_to_first[-2] != 0:
        return
    else:
        first_torch = 0
        for i in range(len(diff_to_first) - 10, len(diff_to_first) - 2):
            if diff_to_first[i : i + 2] == [0, 0] and diff_to_first[i + 2] != 0:
                first_torch = i
                break
        if first_torch == 0:
            return
    # elif diff_to_first[-5] != 0:
    #    return
    changes = changes[: first_torch + 4]

    # save files
    saveFiles(paths[0], output_dir, out_name, fps, frames, [changes[0], changes[-1]])


if __name__ == "__main__":

    # queue = [("demos/MineRLBasaltMakeWaterfall-v0/", SplitWaterfallDemo, "segments/MakeWaterfall/")]
    # queue = [("demos/MineRLBasaltFindCave-v0/", SplitFindCaveDemo, "segments/FindCave/")]
    # queue = [("demos/MineRLBasaltCreateVillageAnimalPen-v0/", SplitCreateVillageAnimalPenDemo, "segments/CreateVillageAnimalPen/")]
    # queue = [("demos/MineRLBasaltBuildVillageHouse-v0/", SplitBuildVillageHouseDemo, "segments/BuildVillageHouse/")]
    # queue = [("/home/aicrowd/data/MineRLBasaltFindCave-v0/", SplitFindCaveDemo, "/home/aicrowd/data/segments/FindCave/")]

    # queue item format (path_to_demos, splitting_method, target_path_for_segments)
    queue = [
        (
            "/home/aicrowd/data/segments/FindCave/stage_2",
            SplitFindCaveDemo,
            "/home/aicrowd/data/segments/FindCave2",
        )
    ]

    # iterate over demos
    for path, method, out_path in queue:
        dir_list = os.listdir(path)
        queue = sorted([x for x in dir_list if x.endswith(".mp4")])

        if method == SplitBuildVillageHouseDemo:
            prev = ""
            new_queue = []
            for filename in queue:
                prefix = "-".join(filename.split("-")[:-2])
                if prefix == prev:
                    new_queue[-1].append(filename)
                else:
                    new_queue.append([filename])
                prev = prefix

            for filenames in tqdm(new_queue, desc="Splitting demos"):
                method(
                    [f"{path}/{filename}" for filename in filenames],
                    out_path,
                    filenames[0],
                )
        else:
            for filename in tqdm(queue, desc="Splitting demos"):
                method(f"{path}/{filename}", out_path, filename)
