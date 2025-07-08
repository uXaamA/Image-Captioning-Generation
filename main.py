# Importing required libraries
import argparse
import wandb
from training import train_model
from inference import CaptionGenerator
from data_util import CaptionDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--wandb_project', default='image-captioning-DL-Assignment 3')
    parser.add_argument('--wandb_entity', required=True)
    parser.add_argument('--model_path', default='best_model.pth')
    parser.add_argument('--image_path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=5e-4)

    args = parser.parse_args()
    # args = parser.parse_args(['--dataset_path', '/path/to/your/dataset',
    #                           '--mode', 'train', # or 'test'
    #                           '--wandb_entity', 'your_wandb_entity'])
    
    if args.mode == 'train':
        train_model(
            args.dataset_path,
            args.wandb_project,
            args.wandb_entity,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    elif args.mode == 'test':
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        dataset = CaptionDataset(args.dataset_path, "", transform=None)
        generator = CaptionGenerator(args.model_path, dataset.vocab)
        caption = generator.generate_caption(args.image_path)
        print(f"Generated Caption: {caption}")
        wandb.finish()

if __name__ == '__main__':
    main()

