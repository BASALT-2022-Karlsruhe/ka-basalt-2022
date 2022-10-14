# Script for segmentation of demos
import cv2
import os
from tqdm import tqdm
import numpy as np

# splits and saves segments based on keyframes
def saveFiles(path, output_dir, filename, fps, frames, keyframes):

    # check for and create dirs
    for i in range(len(keyframes)+1):
        p = f"{output_dir}/stage_{i+1}"
        if not os.path.exists(p):
            os.makedirs(p)

    # write files
    keyframes = [0, *keyframes, -1]
    for i, f in enumerate(keyframes):
        if f == 0:
            continue

        # write video
        video_out = cv2.VideoWriter(f"{output_dir}/stage_{i}/{filename}", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]), 1)
        for frame in frames[keyframes[i-1]:keyframes[i]]:
            video_out.write(frame)

        # write actions jsonl
        lines = open(f"{path[:-4]}.jsonl",'r').readlines()
        with open(f"{output_dir}/stage_{i}/{filename[:-4]}.jsonl",'w') as file:
            for line in lines[keyframes[i-1]:keyframes[i]]:
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
        mean_color = image[344:348,237:243,:].reshape((24,3)).mean(0)

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
    saveFiles(path, output_dir, filename, fps, frames, [keyframe])

# Split FindCave demos into 2 stages based on time. Last 10 seconds are approach, everything before is exploration.
def SplitFindCaveDemo(path, output_dir, out_name):

    last_n_seconds = 10

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

    # collect frames
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if not success:
            break

        frames.append(image)
        count += 1

    # abort if demo reaches maximum length
    if len(frames) >= 3600:
        return
    #print(len(frames))
    keyframe = int(len(frames) - last_n_seconds*fps)

    saveFiles(path, output_dir, filename, fps, frames, [keyframe])


# Splits CreateVillageAnimalPen demos into 3 stages. prior to construction (find place and sometimes collect animals), construction, and bring animals to pen
def SplitCreateVillageAnimalPenDemo(path, output_dir, out_name):
    SplitInventoryChangeDemo(path, output_dir, out_name, slots=[0, 1], max_len=6000)

# Splits BuildVillageHouse demos into 3 stages. prior to construction (find place), construction, and house tour
def SplitBuildVillageHouseDemo(path, output_dir, out_name):
    SplitInventoryChangeDemo(path, output_dir, out_name, slots=[2,3,4,5,6,7,8,9], max_len=14400)

# Split demos into 3 stages based on inventory changes, slots defines which item slots to watch.
def SplitInventoryChangeDemo(path, output_dir, out_name, slots=[0, 1], max_len=6000):
    
    # Path to video file
    vidObj = cv2.VideoCapture(path)
  
    # get FPS
    fps = vidObj.get(cv2.CAP_PROP_FPS)

    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1

    keyframe = -1

    # "framebuffer"
    frames = []

    # stores first item config
    first = None

    # stores last/previous item config
    last = None

    # timestamps of inventory changes
    changes = []

    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if not success:
            break

        frames.append(image)
        count += 1

        if count < 10:
            # skip first 10 frames (only in analysis)
            continue


        # detect and skip crafting menu
        cv2.imwrite("frames/frame%d.jpg" % count, image)
        if np.sum(np.abs(image[110:155,310:327, :].reshape((765,3)).mean(0) - np.array([197, 197, 197]))) < 5 or np.sum(np.abs(image[160:178,370:400,:].reshape((540,3)).mean(0) - np.array([197, 197, 197]))) < 5 or np.sum(np.abs(image[102:120,380:398,:].reshape((324,3)).mean(0) - np.array([197, 197, 197]))) < 5:
            continue

        # switch to black and white as we are only checking for changes in item numbers!
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 220, 255, cv2.THRESH_BINARY)


        if first is None:
            # save first item config for each slot
            first = []
            for slot in slots:
                first.append(blackAndWhiteImage[350:357,237+20*slot:248+20*slot])
        else:
            # calculate diff for each slot w.r.t. initial config
            diff = 0
            for i, slot in enumerate(slots):
                diff += np.sum(np.abs(cv2.subtract(first[i], blackAndWhiteImage[350:357,237+20*slot:248+20*slot])))

            # reset in case we are back to initial config (this is mainly to filter out noise)
            if diff == 0 and keyframe != -1:
                keyframe = -1

            # check for change over initial config
            if diff > 1000:
                # change detected
                if keyframe == -1:
                    # log timestamp of change
                    keyframe = count
                    changes.append(count)

                    # save new config               
                    last = []
                    for slot in slots:
                        last.append(blackAndWhiteImage[350:357,237+20*slot:248+20*slot])
                    
                # get current config (this is done because the diff calculation cv2.subtract will change the image)
                current = []
                for slot in slots:
                    current.append(blackAndWhiteImage[350:357,237+20*slot:248+20*slot])

                # calculate diff with previous config
                diff = 0
                for i, slot in enumerate(slots):
                    diff += np.sum(np.abs(cv2.subtract(last[i], blackAndWhiteImage[350:357,237+20*slot:248+20*slot])))

                if diff > 1000:
                    # next change detected
                    # save config
                    last = current
                    # log timestamp for change
                    changes.append(count)


    # abort if demo reaches maximum length or not enough changes were detected for segmentation
    if len(frames) >= max_len or len(changes) < 2:
        return

    # save files
    saveFiles(path, output_dir, filename, fps, frames, [changes[0], changes[-1]])

  
if __name__ == '__main__':

    #queue = [("demos/MineRLBasaltMakeWaterfall-v0/", SplitWaterfallDemo, "segments/MakeWaterfall/")]
    #queue = [("demos/MineRLBasaltFindCave-v0/", SplitFindCaveDemo, "segments/FindCave/")]
    #queue = [("demos/MineRLBasaltCreateVillageAnimalPen-v0/", SplitCreateVillageAnimalPenDemo, "segments/CreateVillageAnimalPen/")]
    #queue = [("demos/MineRLBasaltBuildVillageHouse-v0/", SplitBuildVillageHouseDemo, "segments/BuildVillageHouse/")]
    
    # queue item format (path_to_demos, splitting_method, target_path_for_segments)
    queue = [
        ("demos/MineRLBasaltMakeWaterfall-v0/", SplitWaterfallDemo, "segments/MakeWaterfall/"),
        ("demos/MineRLBasaltFindCave-v0/", SplitFindCaveDemo, "segments/FindCave/"),
        ("demos/MineRLBasaltCreateVillageAnimalPen-v0/", SplitCreateVillageAnimalPenDemo, "segments/CreateVillageAnimalPen/"),
        ("demos/MineRLBasaltBuildVillageHouse-v0/", SplitBuildVillageHouseDemo, "segments/BuildVillageHouse/"),
        ]

    # iterate over demos
    for path, method, out_path in queue:
        dir_list = os.listdir(path)
        queue = [x for x in dir_list if x.endswith(".mp4")]

        for filename in tqdm(queue, desc="Splitting demos"):
            method(f"{path}/{filename}", out_path, filename)
