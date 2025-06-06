import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, TensorDataset, random_split
#!pip install -U datasets huggingface_hub fsspec #run this if there's an error in dataset loading
from datasets import load_dataset, Dataset
from itertools import batched
from crossable_transformer import CrossableTransformer
from memory_gater import MemoryGater
from latent_embedder import LatentEmbedder
from simple_decoder import SimpleDecoder
from example_model import ExampleModel
from tqdm import tqdm
from torchviz import make_dot


def create_new_token_data(tensor, seq_len):
    total_len = tensor.shape[0]
    n = total_len - seq_len

    if seq_len >= total_len:
        x = None
        y = None
    else:
        x = [tensor[i:i+seq_len] for i in range(n)]
        y = [tensor[i+seq_len] for i in range(n)]

    return x, y

def main():
    token_dim = 768
    emb_dim = 256
    hidden_dim = 128
    m_dim = 128
    n_stacks = 3
    n_heads = 8
    dropout = 0.1
    dict_size = 16384

    batch_size = 32
    epochs = 10
    static_seq = 64




    encoder = nn.Sequential(nn.Embedding(dict_size, token_dim), LatentEmbedder(token_dim, emb_dim, hidden_dim=hidden_dim))
    model = ExampleModel(encoder, m_dim, token_dim, emb_dim, dict_size, hidden_dim, n_stacks, dropout, n_heads)



    '''
    dot = make_dot(model(toy_input), params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('architecture')
    '''

    #TODO model doesn't accept differently sized batches. Because the gates and alloc absorb the batch size
    '''
    toy_input = torch.randint(low=0, high=dict_size, size=(32, static_seq), dtype=torch.long)
    toy_input2 = torch.randint(low=0, high=dict_size, size=(64, static_seq+5), dtype=torch.long)

    print(model(toy_input),model(toy_input2))
    '''

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=dict_size)
    tokenizer.train_from_iterator(" ".join(dataset["text"]), trainer)


    x, y  = [], []
    for text in dataset["text"]:
        if len(text) > 0:
            x_t, y_t = create_new_token_data(torch.tensor(tokenizer.encode(text).ids), static_seq)
            if x_t is not None:
                x.append(x_t)
                y.append(y_t)

    flat_x = ([item for sublist in x for item in sublist])
    flat_y = ([item for sublist in y for item in sublist])



    x = list(batched(flat_x, batch_size))[:-1] #I don't care calling it x again
    y = list(batched(flat_y, batch_size))[:-1]
    data = list(zip(x, y))

    device = torch.device("cuda")

    flat_x, flat_y = torch.stack(flat_x), torch.stack(flat_y)


    dataset = TensorDataset(flat_x, flat_y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)#, pin_memory=True)



    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    token_loss = nn.CrossEntropyLoss()



    del x, y, flat_x, flat_y#, x_shuffled, y_shuffled



    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


    model = model.to(device)



    for epoch in range(epochs):
        loss_1_t = 0
        loss_2_t = 0
        model.reset_memory()
        model.train()
        mem_alloc = torch.cuda.memory_allocated(device)/ 1024**3
        for name, param in model.named_buffers():
            if param.requires_grad:
                print(f"Buffer {name} requires grad, if it's memory then this is an error")

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))#, desc=f"Latent loss (av):{loss_1_t}, token_loss (av): {loss_2_t}, memory alloc (GiB):{mem_alloc}", leave=False)

        for i, (inputs, label) in enumerate(train_loader):
            inputs = inputs.to(device , non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            label_temp = label.unsqueeze(1)
            new_inputs = torch.cat([inputs, label_temp], 1)

            latent, out_temp = model(inputs)
            outputs = model.logits(latent.detach())
            target_latent = model.encoder(new_inputs[:, 1:])

            cos_sim = F.cosine_similarity(latent, target_latent, dim=-1)
            smoothL1 = F.smooth_l1_loss(latent,target_latent)
            loss_latent = (1 - cos_sim.mean())*0.5 + (smoothL1.mean())*0.5

            loss_t = token_loss(outputs[:,-1,:], label) # Decide between choosing last of these or applying commented mean and model outputs 1 token

            (loss_latent + loss_t).backward(retain_graph=False) #True

            optimizer.step()

            model.memory = model.memory.detach() #Do this if you don't want to explode in tens of gigabytes used by torch
            #model.memory = new_memory_values.clone().detach().requires_grad_(True)   #???

            loss_1_t = (loss_1_t * i + loss_latent) / (i + 1)
            loss_2_t = (loss_2_t * i + loss_t) / (i + 1)

            progress_bar.set_postfix(loss_1_t=loss_1_t.item(), loss_2_t=loss_2_t.item(), mem_alloc= torch.cuda.memory_allocated(device) / 1024**3)
            progress_bar.update(1)

        model.eval()
        v_loss_1_t = 0
        v_loss_2_t = 0
        with torch.no_grad():
            for i, (inputs, label) in enumerate(test_loader):

                inputs = inputs.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                label_temp = label.unsqueeze(1)
                new_inputs = torch.cat([inputs, label_temp], 1)

                latent, outputs = model(inputs)

                target_latent = model.encoder(new_inputs[:, 1:])

                v_loss_latent = (1 - F.cosine_similarity(latent, target_latent, dim=-1).mean())*0.5 + (F.smooth_l1_loss(latent, target_latent, dim=-1).mean())*0.5
                v_loss_t = token_loss(outputs, label)

                v_loss_1_t = (v_loss_1_t * i + v_loss_latent) / (i + 1)
                v_loss_2_t = (v_loss_2_t * i + v_loss_t) / (i + 1)

        print(f"Epoch {epoch + 1}: latent loss (av):{loss_1_t}, token_loss (av): {loss_2_t}, val latent loss (av):{v_loss_1_t}, val_token_loss (av):{v_loss_2_t}, memory alloc (GiB): {(torch.cuda.memory_allocated(device)/ 1024**3)}")


if __name__ == '__main__':
    main()