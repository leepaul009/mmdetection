import pandas as pd
import torch, os, json
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input", type=str, default="pred.pth")
parser.add_argument('--output', type=str, default="output.csv")
parser.add_argument('--score', type=float, default=0.5)
args = parser.parse_args()

res_path = args.input
to_path  = args.output
score_threshold = args.score

results = torch.load(res_path)

#image_id,score,x,y,width,height
data_dict = {
    "image_id": [],
    "score": [],
    "x": [],
    "y": [],
    "width": [],
    "height": [],
}

for res in results:
    #if res['score'] < score_threshold:
    #    continue

    x,y,w,h = list(map(int, res['bbox']))

    #if h < 25:
    #    continue

    data_dict["image_id"].append(res['image_id'])
    data_dict["score"].append(res['score'])
    data_dict["x"].append(x)
    data_dict["y"].append(y)
    data_dict["width"].append(w)
    data_dict["height"].append(h)


df = pd.DataFrame(data_dict)
df.to_csv(to_path, index=False)
print("save output as {}".format(to_path))

