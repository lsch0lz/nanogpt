import torch
import argparse

from dataloader import DataLoader
from model import NanoGPTModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--file_path", type=str, default="./input.txt")
    parser.add_argument("--train_split", type=float, default=0.9)

    return parser.parse_args()


def estimate_loss_model(_model, num_eval_iter: int, dataloader: DataLoader):
    output_dict: dict = {}
    _model.eval()

    for split in ["train", "val"]:
        _losses: torch.Tensor = torch.zeros(num_eval_iter)
        for k in range(num_eval_iter):
            x, y, _ = dataloader.create_batches(split)

            _logits, _loss = _model(x, y)
            _losses[k] = _loss.item()

        output_dict[split] = _losses.mean()

    _model.train()
    return output_dict


def train_loop(max_iters: int, eval_interval: int, device):
    for iteration in range(max_iters):

        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            losses = estimate_loss_model(model, num_eval_iter=500, dataloader=dataloader)
            print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb, _ = dataloader.create_batches('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decoder(model.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    args = parse_args()

    dataloader: DataLoader = DataLoader(
        file_path=args.file_path,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_eval_iter=args.eval_iters,
        train_split=args.train_split
    )

    _, decoder, _ = dataloader.create_token_encoder_and_decoder()
    _, _, vocab_size = dataloader.create_batches("train")

    model = NanoGPTModel(
        vocab_size=vocab_size,
        num_embeddings=args.n_embd,
        block_size=args.block_size,
        num_head=args.n_head,
        num_layer=args.n_layer,
        dropout=args.dropout,
        device=args.device
    )

    model = model.to(args.device)

    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_loop(args.max_iters, args.eval_interval, args.device)
