# KA BASALT

## Setting up

Download the dummy BASALT dataset from [here](https://microsofteur-my.sharepoint.com/:f:/g/personal/t-anssik_microsoft_com/Ej9R17fChVVLtPZQmnA233ABmhtzPBnS-v0BOv6na8_IZA?e=izua7z) (password: `findcave2022`). Also download the 1x width foundational model [.weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights) and [.model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model) files for the OpenAI VPT model.
Place these data files under `data` to match the following structure:

```
├── data
│   ├── MineRLBasaltBuildVillageHouse-v0
│   │   ├── Player70-f153ac423f61-20220707-111912.jsonl
│   │   ├── Player70-f153ac423f61-20220707-111912.mp4
│   │   └── ... rest of the files
│   ├── MineRLBasaltCreateVillageAnimalPen-v0
│   │   └── ... files as above
│   ├── MineRLBasaltFindCave-v0
│   │   └── ... files as above
│   ├── MineRLBasaltMakeWaterfall-v0
│   │   └── ... files as above
│   └── VPT-models
│       ├── foundation-model-1x.model
│       └── foundation-model-1x.weights
```

## Build & Start Docker Container
`sh run_docker.sh`

train.py is executed after building & starting the container
This will save a fine-tuned network for each task under `train` directory. 
This has been tested to fit into a 8GB GPU.

## Visualizing/enjoying/evaluating models (TODO: test in docker container)
To run the trained model for `MineRLBasaltFindCave-v0`, run the following:

```
python run_agent.py --model data/VPT-models/foundation-model-1x.model --weights train/MineRLBasaltFindCave.weights --env MineRLBasaltFindCave-v0 --show
```

Change `FindCave` to other tasks to run for different tasks.

## How to Submit a Model on AICrowd.

To submit this baseline agent follow the [submission instructions](https://github.com/minerllabs/basalt_2022_competition_submission_template/), but use this repo instead of the starter kit repo.
