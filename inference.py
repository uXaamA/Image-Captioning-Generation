# importing required libraries
import torch
from torchvision import transforms
from PIL import Image
import wandb
from model import ImageCaptionModel
from torch.cuda.amp import autocast

class CaptionGenerator:
    def __init__(self, model_path, vocab, device='cuda'):
        self.device = device
        self.vocab = vocab
        self.model = ImageCaptionModel(len(vocab)).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ])

    def generate_caption(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        caption = [self.vocab.stoi['<sos']]

        with torch.no_grad(), autocast():
            features = self.model.encoder(image).unsqueeze(1)

            for _ in range(30): # max length
                inputs = torch.tensor(caption).unsqueeze(0).to(self.device)
                embeddings = self.model.embed(inputs)
                outputs = torch.cat([features, embeddings], dim=1)
                out, _ = self.model.lstm(outputs)
                pred = self.model.fc(out[:, -1, :])
                next_word = pred.argmax().item()

                if next_word == self.vocab.stoi['<eos>']:
                    break
                caption.append(next_word)

        # log to wandb
        caption_words = [self.vocab.itos[idx] for idx in caption]
        wandb.log({
            'input_image' : wandb.Image(image_path),
            'generated_caption' : ' '.join(caption_words[1: -1])
        })
        return ' '.join(caption_words[1: -1])

