import torch
import torch.nn as nn
from torch.nn import functional as F

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        #
        # nn.Embedding is a lookup table that stores embeddings of a fixed dictionary and size.
        # It takes two arguments:
        # 1. the size of the dictionary of embeddings: vocab_size  ; here we have 65 different tokens
        # 2. the size of each embedding vector: embed_dim          ; how many numbers are used to describe each token
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)

    # taking the input tensor and returning the embedded tensor
    def forward(self,input_context):
        embedding_table=self.token_embedding_table(input_context)
        #
        print(f'input_contex.shape={input_context.shape}')
        print(f'embedding_table.shape={embedding_table.shape}')

        return embedding_table


class SelfAttention(nn.Module):
    def __init__(self,embed_dim):
        super().__init__()

        ### 1) q, k, v linear transformation matrices
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self,context):
        q = self.query(context) # dimension: (B, T, C)
        k = self.key(context)   # dimension: (B, T, C)
        v = self.value(context) # dimension: (B, T, C)

        ### 2) Q @ K^T
        # We need to keep the first dimension to be the batch size: "B"
        # PyTorch will be able to perform matmul in parallel (at different batch indices)
        # We need to transpose the other two indices (-2 and 01)

        print('q.shape,k.shape=',q.shape,k.shape)
        k_transposed = k.transpose(-2, -1) #swaps the dim "-2" with dim "-1"
        print('k_transposed.shape=',k_transposed.shape)

        ### 2-3) scaling the attention scores
        embed_dim = q.shape[-1]
        scores = q @ k_transposed * (embed_dim ** -0.5)  # Scale by sqrt(d_k) # [B, T, T]

        ### 4) Masking
        # We want to mask the next tokens from affecting the current token
        # at i, we have to mask j>i --> similar to a lower triangular matrix
        # We set them to -inf so after softmax, they become 0

        print('---')
        print('scores at B=0, before masking: \n', scores[0,:,:],'\n ---')
        seq_size = q.shape[-2]
        self.tril=torch.tril(torch.ones(seq_size, seq_size, device=scores.device))

        scores = scores.masked_fill(
            self.tril[:seq_size, :seq_size] == 0, float('-inf')
            ) # (B, T, T)
        print('scores at B=0, after masking: \n', scores[0,:,:],'\n ---')

        ### 5) applying the softmax
        # scores has the dimension [B, T, T]
        # we want to normalize the score for each query position independently
        # so, softmax is applied along the last dimension

        scores = F.softmax(scores, dim=-1) #[B, T, T]
        print('scores at B=0, after softmax: \n', scores[0,:,:],'\n ---')

        ### 6) multiplying by v matrix: [B, T, C]
        scores = scores @ v

        return scores # B, T, C]


class SingleHeadAttention(nn.Module):
    def __init__(self,embed_dim,head_size, masked):
        super().__init__()
        self.masked = masked
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key   = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

    def forward(self,context):
        q = self.query(context) # dimension: [B, T, C']
        k = self.key(context)   # dimension: [B, T, C']
        v = self.value(context) # dimension: [B, T, C']

        seq_size  = q.shape[-2]
        embed_dim = q.shape[-1]

        k_transposed = k.transpose(-2, -1) #swaps the dim "-2" with dim "-1"
        scores = q @ k_transposed * (embed_dim ** -0.5)  # Scale by sqrt(d_k) # [B, T, T]

        if self.masked:
            self.tril=torch.tril(torch.ones(seq_size, seq_size, device=scores.device))
            scores = scores.masked_fill(
                self.tril[:seq_size, :seq_size] == 0, float('-inf')
                ) # (B, T, T)

        scores = F.softmax(scores, dim=-1) #[B, T, T]
        if not self.training:
            self.last_weights = scores.detach()
        
        scores = scores @ v #[B, T, T] x [B, T, C'] = [B, T, C']
        return scores #[B, T, C']


class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,n_heads,dropout, masked):
        super().__init__()

        # 3a) Creating a list of SingleHeadAttention
        # Not just a simple list, but a ModuleList.
        # It allows the model to track the parameters
        if n_heads <= 0 or embed_dim % n_heads != 0:
            raise ValueError("embed_dim must be divisible by a positive n_heads")
        head_size = embed_dim//n_heads
        self.heads = nn.ModuleList([SingleHeadAttention(embed_dim,head_size,masked=masked)
                                for _ in range(n_heads)])

        # 4) Adding an additional linear layer (called projection)
        # Will take the concatanated attentions [B,T,C]
        # Its output should have the same dimension as the embedded context [B, T, C]
        self.proj  = nn.Linear(embed_dim,embed_dim)

        # 5) Adding the residual with dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self,context):

        # 3b) concatanating the list of SingleHeads modules
        heads_list=[]
        self.last_weights_4_heads = []
        for head in self.heads:
            heads_list.append(head(context))

            if not self.training and len(self.last_weights_4_heads) < 4:
                self.last_weights_4_heads.append(head.last_weights)

        heads = torch.cat(heads_list,dim=-1)
        #print(f'heads.shape={heads.shape}\n ---')

        # 4) adding the linear layer, called projection
        scores=self.proj(heads) # now score is [B, T, C]

        # 5) adding the dropout layer
        scores = self.dropout(scores)

        return scores #[B, T, C]


class FeedForward(nn.Module):
    def __init__(self,embed_dim,dropout):
        super().__init__()

        # The input to the network is [B, T, C]
        self.network=nn.Sequential(
            # 1) Increasing the dimension to [B, T, 4C]
            nn.Linear(embed_dim, 4*embed_dim),
            #
            # 2) ReLU
            nn.ReLU(),
            #
            # 3) decreasing the dimension back to [B, T, C]
            nn.Linear(4*embed_dim, embed_dim),
            #
            # 4) Adding the dropout
            nn.Dropout(dropout),
        )

    def forward(self, scores):
        return self.network(scores)


class AttentionBlock(nn.Module):
    def __init__(self,embed_dim,n_heads,dropout,masked):
        super().__init__()

        # 1) Instantiating the MHA
        # Input : [B, T, C]
        # Output: [B, T, C]
        self.mha = MultiHeadAttention(embed_dim,n_heads,dropout,masked=masked)

        # 2) Creating the layer normalization module 1
        # Input: Embedded context [B, T, C]
        # Output: will be summed up with the MHA output [B, T, C]
        self.ln1 = nn.LayerNorm(embed_dim)

        # 3) Instantiating the FF
        self.ff = FeedForward(embed_dim,dropout)

        # 4) Creating the layer normalization module 2
        # Input: Output of MHA [B, T, C]
        # Output: will be summed up with the FF output [B, T, C]
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self,x):
        # 5) Connecting layer norm 1 and MHA
        x = x + self.mha(self.ln1(x))

        #6) Connecting layer norm 2 and FeedForward
        return x + self.ff(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self,vocab_size,seq_size,embed_dim,n_heads,n_layers,dropout, positional_embedding=True, masked=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_size = seq_size

        # 1) Instantiating Token embedding
        self.TokenEmbedding = nn.Embedding(vocab_size, embed_dim) # output: [B,T,C]

        # 2) Instantiating Positional embedding
        self.pm = positional_embedding
        self.PositionalEmbedding = nn.Embedding(seq_size, embed_dim) #output: [T,C]

        # 3) Instantiating Attention block
        # Creating a list of Block modules
        block_list = []
        for _ in range(n_layers):
            block = AttentionBlock(embed_dim, n_heads, dropout, masked=masked)
            block_list.append(block)
        # Convert the list into a Sequential container
        self.blocks = nn.Sequential(*block_list) # * unpacks the list

        # 4) Last Layer Normalization module
        self.lnn = nn.LayerNorm(embed_dim)

        # 5) Last Linear layer to go from [B,T,C] to [B,T,vocan_size]
        self.linearn = nn.Linear(embed_dim,vocab_size)

    def forward(self,context, targets=None):

        device = self.TokenEmbedding.weight.device
        context = context.to(device)
        if targets is not None: targets = targets.to(device)

        batch_size = context.shape[0]
        seq_size   = context.shape[1]


        # 1) Token embedding
        tm = self.TokenEmbedding(context) # [B, T, C]

        # 2) Positional embedding
        # create one position index for each token in the context
        tmp = torch.arange(seq_size, dtype=torch.int64, device=device)
        pm  = self.PositionalEmbedding(tmp)

        # Adding the two embeddings
        x = tm + pm if self.pm else tm

        # 3) Calculate the blocks
        x = self.blocks(x)

        # 4 and 5) Pass the data through lnn and last linear layer
        x = self.lnn(x)
        y = self.linearn(x) # output: [B, T, vocab_size]

        if not self.training:
            self.last_weights = self.blocks[0].mha.last_weights_4_heads

        # If there are targets, it's training otherwise is inference
        if targets is None:
            loss = None
        else:
            y       = y.reshape(batch_size*seq_size, self.vocab_size)
            targets = targets.reshape(batch_size*seq_size)
            loss    = F.cross_entropy(y, targets)

        return y, loss

    def generation(self, context, max_tokens):
        # context has dimensions of [B, T]
        context = context.to(self.TokenEmbedding.weight.device)
        for _ in range(max_tokens):
            # make sure the context fits in the sequence length
            context_crop = context[:, -self.seq_size:]

            # get the predictions
            y, loss = self(context_crop)

            # focus only on the last token
            y = y[:, -1, :] # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(y, dim=-1) # (B, C)

            # sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append the sample to the running sequence
            context = torch.cat((context, next_token), dim=1) # (B, T+1)
        return context
