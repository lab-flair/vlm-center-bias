from argparse import ArgumentParser
import os
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import open_clip
import clip


def main(args):
    print(args)
    # Load model and tokenizer
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    if args.model_name == "negclip":
        path = "./checkpoints/negclip.pth"
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            import gdown
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    model.cuda().eval() 
    tokenizer = open_clip.get_tokenizer(args.model_name) if not args.model_name == "negclip" else clip.tokenize

    choices = [0, 0, 0, 0]
    on_total, other_total, on_correct, other_correct = 0, 0, 0, 0

    # Extract images and ground truth labels from filenames
    for filename in tqdm(os.listdir(args.image_dir)):
        if filename.endswith(".jpeg"):
            # Extract objects from filename
            parts = filename.split("_")
            if len(parts) >= 3:
                object1, object2 = parts[0].replace("-", " "), parts[-1].replace("-", " ")
                object2 = object2.replace(".jpeg", "")
                # print(f"object1: {object1}, object2: {object2}")
                image = preprocess(Image.open(os.path.join(args.image_dir, filename))).cuda().unsqueeze(0)
                label = f"a {object1} and a {object2}"
                text = tokenizer([
                    label,  # Exact label
                    f"a {object1}",  # Only object1
                    f"a {object2}",  # Only object2
                    "a cat and a refrigerator",  # Arbitrary objects
                ]).cuda()

                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).tolist()[0]
                    
                if "_on_" in filename:
                    on_correct += 1 if int(np.argmax(text_probs)) == 0 else 0
                    on_total += 1
                else:
                    other_correct += 1 if int(np.argmax(text_probs)) == 0 else 0
                    other_total += 1
                choices[np.argmax(text_probs)] += 1
    print(f"overall accuracy = {(on_correct + other_correct) / (on_total + other_total)}")
    print(f"center accuracy = {on_correct / on_total}")
    print(f"off-center accuracy = {other_correct / other_total}")
    print(f"stats of choices = {choices}")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--image_dir", type=str, default="./data/whatsup_images")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_arguments())