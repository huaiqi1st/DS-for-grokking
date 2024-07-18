import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import einops
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass(frozen = True)
class Config():
    lr: float = 1e-3
    weight_decay: float = 1.0
    p: int = 97
    d_model: int = 128
    fn_name: str = 'add'
    frac_train: float = 0.3
    num_epochs: int = 50000
    save_models: bool = True
    save_every: int = 100
    stopping_thresh: int = -1
    seed: int = 0
    num_layers: int = 1
    batch_style: str = 'full'
    d_vocab: int = 98
    n_ctx: int = 3
    d_mlp: int = 4*d_model
    num_heads: int = 4
    act_type: str = 'ReLU'
    device: t.device = t.device("cuda")
    use_ln: bool = False
    take_metrics_every_n_epochs: int = 1000
    l2_lambda: float = 1e-5  # L2 regularization coefficient

    @property
    def d_head(self):
        return self.d_model // self.num_heads

    @property
    def random_answers(self):
        return np.random.randint(low=0, high=self.p, size=(self.p, self.p))

    @property
    def fns_dict(self):
        return {
            'add': lambda x, y: (x + y) % self.p,
            'subtract': lambda x, y: (x - y) % self.p,
            'x2xyy2': lambda x, y: (x**2 + x * y + y**2) % self.p,
            'rand': lambda x, y: self.random_answers[x][y]
        }

    @property
    def fn(self):
        return self.fns_dict[self.fn_name]

    def is_it_time_to_take_metrics(self, epoch):
        return epoch % self.take_metrics_every_n_epochs == 0

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return (x @ self.W_U)

class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]

class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis[-1])[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = t.einsum('ihd,bpd->biph', self.W_K, x)
        q = t.einsum('ihd,bpd->biph', self.W_Q, x)
        v = t.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        z = t.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        assert act_type in ['ReLU', 'GeLU']

    def forward(self, x):
        x = t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        if self.act_type == 'ReLU':
            x = F.relu(x)
        elif self.act_type == 'GeLU':
            x = F.gelu(x)
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config: Config, use_cache=False, use_ln=True):
        super().__init__()
        self.embed = Embed(d_vocab=config.d_vocab, d_model=config.d_model)
        self.pos_embed = PosEmbed(max_ctx=config.n_ctx, d_model=config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=config.d_model,
                                                      d_mlp=config.d_mlp,
                                                      d_head=config.d_head,
                                                      num_heads=config.num_heads,
                                                      n_ctx=config.n_ctx,
                                                      act_type=config.act_type,
                                                      model=[self]) for i in range(config.num_layers)])
        self.unembed = Unembed(d_vocab=config.d_vocab, d_model=config.d_model)
        self.use_ln = use_ln

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x

def gen_train_test(config: Config):
    num_to_generate = 3000
    pairs = [(random.randint(0, 97), random.randint(0, 97), config.p) for _ in range(num_to_generate)]
    random.seed(config.seed)
    random.shuffle(pairs)
    div = int(config.frac_train * len(pairs))
    return pairs[:div], pairs[div:]

def prepare_data(pairs):
    inputs = t.tensor([(i, j, p) for i, j, p in pairs], dtype=t.long).to(pairs[0][2])
    return inputs

def full_loss(config: Config, model: Transformer, data):
    inputs = prepare_data(data)
    logits = model(inputs)[:, -1]
    labels = t.tensor([config.fn(i, j) for i, j, _ in data]).to(config.device)
    ce_loss = F.cross_entropy(logits, labels)

    # L2 regularization (weight decay)
    l2_loss = 0
    for param in model.parameters():
        l2_loss += t.sum(param.pow(2))

    total_loss = ce_loss + config.l2_lambda * l2_loss
    return total_loss

def calculate_accuracy(config: Config, model: Transformer, data):
    inputs = prepare_data(data)
    logits = model(inputs)[:, -1]
    labels = t.tensor([config.fn(i, j) for i, j, _ in data]).to(config.device)
    predictions = logits.argmax(dim=-1)
    correct_predictions = (predictions == labels).sum().item()
    accuracy = correct_predictions / len(labels)
    return accuracy

class Trainer:
    def __init__(self, config: Config, model=None) -> None:
        self.model = model if model is not None else Transformer(config, use_cache=False)
        self.model.to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step / 10, 1))
        self.train, self.test = gen_train_test(config=config)
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.config = config

    def do_a_training_step(self, epoch: int):
        train_loss = full_loss(config=self.config, model=self.model, data=self.train)
        test_loss = full_loss(config=self.config, model=self.model, data=self.test)
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())
        train_accuracy = calculate_accuracy(config=self.config, model=self.model, data=self.train)
        test_accuracy = calculate_accuracy(config=self.config, model=self.model, data=self.test)
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, train loss {train_loss.item():.4f}, test loss {test_loss.item():.4f}, train accuracy {train_accuracy:.4f}, test accuracy {test_accuracy:.4f}')
        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return train_loss, test_loss

    def post_training_save(self):
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'epoch': self.config.num_epochs,
        }
        t.save(save_dict, f"final.pth")
        print(f"Saved model to {'final.pth'}")

    def plot_metrics(self):
        epochs = range(len(self.train_losses))
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.test_losses, label='Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.test_accuracies, label='Testing Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')
        plt.show()

def train_model(config: Config):
    world = Trainer(config=config)
    print(f'Run name {int(time.time())}')

    for epoch in range(config.num_epochs):
        train_loss, test_loss = world.do_a_training_step(epoch)
        if test_loss.item() < config.stopping_thresh:
            break

    world.post_training_save()
    world.plot_metrics()
    return world


if __name__ == "__main__":
    config = Config()
    trainer = train_model(config)
    torch.save(trainer.model.state_dict(), "trained_model.pth")