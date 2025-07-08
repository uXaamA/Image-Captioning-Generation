import torch 
import torch.nn as nn
import torchvision.models as models
import wandb

class ImageCaptionModel(nn.Module):
    def __init__ (self, vocab_size, embed_size=512, hidden_size=512, num_layers=1):
        super().__init__()
        self.encoder = models.resnet50(pretrained=True)     #image feature extractor
        # replacing the last classification layer of resnet50 to match embed_size of words
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.embed = nn.Embedding(vocab_size, embed_size) # convert words to vector
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # tracking the hyperparameters using wandb
        wandb.config.update({
            'embed_size' : embed_size,
            'hidden_size' : hidden_size, 
            'num_layers' : num_layers,
            'optimizer' : 'Adam',
            'mixed_precision' : True
        })

    def forward(self, images, captions):
        features = self.encoder(images).unsqueeze(1)   # pass the image through the encoder (resnet50) and adding new dimension shape [batch, 1, embed_size]
        embeddings = self.embed(captions)  # converting the captions into embeddings
        embeddings = torch.cat((features, embeddings), dim=1) # concatenating the image features and the words embeddings new shaper[batch, 1 + caption_len, embed_size]
        out, _ = self.lstm(embeddings)  # passing the concatenated features through the lstm
        outputs = self.fc(out)  # passing the lstm output through the fully connected layer
        return outputs

def init_model(vocab_size, device):
    """ Initialize the model and move it to the specified device """
    model = ImageCaptionModel(vocab_size).to(device)  # initializing the model
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model
