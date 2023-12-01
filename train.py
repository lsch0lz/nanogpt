import torch

from dataloader import DataLoader
from model import NanoGPTModel

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------


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


if __name__ == "__main__":
    dataloader: DataLoader = DataLoader(
        file_path="./input.txt",
        block_size=256,
        batch_size=64,
        num_eval_iter=500,
        train_split=0.9
    )

    _, decoder, _ = dataloader.create_token_encoder_and_decoder()
    _, _, vocab_size = dataloader.create_batches("train")

    model = NanoGPTModel(
        vocab_size=vocab_size,
        num_embeddings=384,
        block_size=256,
        num_head=6,
        num_layer=6,
        dropout=0.2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iteration in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            losses = estimate_loss_model(model, num_eval_iter=500, dataloader=dataloader)
            print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb, _ = dataloader.create_batches('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decoder(model.generate(context, max_new_tokens=500)[0].tolist()))
    #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))