import argparse
import os
import sys
import PIL
import torch
import open_clip
from tqdm.auto import tqdm, trange

def load_image(path, i, device):
    path = os.path.join(path, f"{i:05}")
    meta = torch.load(path + ".pt", map_location=torch.device(device),)
    with PIL.Image.open(path + ".png") as img:
        return img.copy(), meta
    
def delim_print(l, delim="\t"):
    print(delim.join([ str(x) for x in l ]))

class CLIPScore:
    def __init__(self, device, model_type, model_pretrain):
        self.model, _, self.clip_preprocess = open_clip.create_model_and_transforms(model_type,
                                                                        pretrained=model_pretrain,
                                                                        device=device)
        self.tokenizer = open_clip.get_tokenizer(model_type)
        self.device = device

    def score(self, images, prompt):
        with torch.no_grad():
            img_batch = [self.clip_preprocess(i).unsqueeze(0) for i in images]
            img_batch = torch.concatenate(img_batch).to(self.device)
            image_features = self.model.encode_image(img_batch)

            text = self.tokenizer([prompt]).to(self.device)
            text_features = self.model.encode_text(text)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            blah = image_features @ text_features.T
            return (image_features @ text_features.T).mean(-1)
        
def print_score_for_path(path, start, end, device):
    scores = []
    for i in trange(start, end):
        img, meta = load_image(path, i, device)
        scores.append(clip_score.score([img], meta["prompt"]))

    scores = torch.tensor(scores)
    delim_print([path, start, end, scores.mean().item(), scores.std().item()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP Score images')
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--model', default="ViT-g-14")
    parser.add_argument('--model_pretrain', default="laion2b_s12b_b42k")
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('path', nargs='+', help="Path to images to score.")
    args = parser.parse_args()

    clip_score = CLIPScore(args.device, args.model, args.model_pretrain)
    for path in tqdm(args.path):
        try:
            print_score_for_path(path, args.start, args.end, args.device)
        except Exception as ex:
            print("Skipping path", path, ex, file=sys.stderr)