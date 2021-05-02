import torch
import torch.nn as nn
import math
from variables_ecog import COLUMN_WIDTH, TRG_PAD_IDX, SRC_PAD_IDX, device
import pdb


class PositionalEncodingEcog(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingEcog, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        dim = x.shape
        for i in range(dim[0]):
            x[0, i, :] = x[0, i, :] + self.pe[0,0, :]
        return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        dim = x.shape
        x = x + self.pe[:dim[0],:dim[1], :]
        return self.dropout(x)


class Conv1d(nn.Module):
    def __init__(self, n_channels, stride):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(n_channels, n_channels, stride, stride=stride)
    
    def forward(self, x):
        return self.conv(x)


class FeedForward(nn.Module):
    def __init__(
        self, 
        d_model=512, 
        d_ff=2048,
        P_drop=0.1,
        d_input=None
        ):
        '''
        :param d_model: This variable is the dimension of the model. In the
                        paper they use 512.
        :param d_ff: This variable represents the dimension of the feed forward
                     network.
        :param P_drop: This is the dropout rate. In the paper they use 0.1.
        '''
        super(FeedForward, self).__init__()

        if d_input == None:
            d_input = d_model
        #  initializes two linear transformations
        self.linear_1 = nn.Linear(d_input, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        #  initializes dropout
        self.dropout = nn.Dropout(P_drop)


    def forward(self, x):
        '''
        :param x: layer(x)
        :return: returns two linear transformations with a ReLu in-between
        '''
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        

class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 d_model=512,
                 N=6,
                 h=8,
                 d_ff=2048,
                 P_drop=0.1,
                 max_length=1000
                 ):
        '''
        :param src_vocab_size: size of the vocab list in dataset
        :param d_model: The dimension of the model.
        :param N: The number of layers the encoder has.
        :param h: The number of heads that the model will have.
        :param d_ff: The dimension of the feed forward network.
        :param P_drop: The dropout rate of the model.
        :param max_length: The maximum length of the input.
        '''
        super(Encoder, self).__init__()

        #  Initializing the input embedding and the positional embedding
        self.PositionalEncodingEcog = PositionalEncoding(d_model)
        # modification to model
        self.norm = nn.LayerNorm(d_model)
        
        # self.feedForward = FeedForward_Mod(COLUMN_WIDTH//12, d_model, d_ff)
        self.feedForward = FeedForward(d_model, d_ff, P_drop, 64)
        #  Creating N layers by calling EncoderLayer N times
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, P_drop) for _ in range(N)])
        #  Initializing dropout
        self.dropout = nn.Dropout(P_drop)
        #  Saving the dimension of model to perform scaling
        self.d_model = d_model
        self.d_ff = d_ff
        #  Initializing convolution
        self.Conv = Conv1d(COLUMN_WIDTH, 12)


    def forward(self, source, mask):
        '''
        :param source: It is the input for the model.
        :param mask: mask for the input.
        :return: The output of all the encoder layers.
        '''
    
        output = self.feedForward(source) * math.sqrt(self.d_model)
        # pdb.set_trace()
        output = self.PositionalEncodingEcog(output)
        for layer in self.layers:
            output = layer(output, mask)
        
        return output

class EncoderLayer(nn.Module):
    def __init__(
        self, 
        d_model=512, 
        h=8, 
        d_ff=2048, 
        P_drop=0.1
        ):
        '''
        :param d_model: The dimension of the model.
        :param h: number of heads it will have. (multi-head attention).
        :param d_ff: dimension of the feed forward network.
        :param P_drop: dropout rate.
        '''
        super(EncoderLayer, self).__init__()

        #  Norm after Masked Multi-Head Self-Attention in the encoder block
        self.norm1 = nn.LayerNorm(d_model)
        #  Norm after the feed forward block in the encoder
        self.norm2 = nn.LayerNorm(d_model)
        # Initializing self attention. It will take in the Inputs(src) after
        # positional embedding.

        # self.self_attention = nn.MultiheadAttention(d_model, h, P_drop)
        self.self_attention = MultiHeadAttention(d_model, h, P_drop)
        
        
        # Initializing FeedForward class
        self.feedForward = FeedForward(d_model, d_ff, P_drop)
        # Initializing dropout
        self.dropout = nn.Dropout(P_drop)
     

    def forward(self, source, mask):
        '''
        :param source: input sequence
        :param mask: mask
        :return: the norm of output of the feed forward network
        '''
        #  self_attention takes in the input(source) for Q, K, V
        # source = source.permute(1,0,2)
        temp, _ = self.self_attention(source, source, source, None)
        # temp = temp.permute(1,0,2)
        # source = source.permute(1,0,2)
        source = self.norm1(source + self.dropout(temp))

        temp = self.feedForward(source)

        return self.norm2(source + self.dropout(temp))

class Decoder(nn.Module):
    def __init__(
        self,
        target_size,
        d_model=512,
        N=6,
        h=8,
        d_ff=2048,
        P_drop=0.1,
        max_length=905
        ):
        '''
        :param target_size: Size of the output(target)
        :param d_model: dimension of the model
        :param N: Number of layers
        :param h: number of heads for multi-headed attention
        :param d_ff: dimension of the feed forward network
        :param P_drop: dropout rate
        :param max_length: maximum sequence length
        '''
        super(Decoder, self).__init__()

        #  Initializing the embedding layers
        self.outputEmbedding = nn.Embedding(target_size, d_model)
        self.positionalEncoding = PositionalEncoding(d_model)
      
        #  Creating N layers
        self.layers = nn.ModuleList([DecoderLayer(d_model, h, d_ff, P_drop) for _ in range(N)])
        #  Initialized dropout
        self.dropout = nn.Dropout(P_drop)
        #  created d_model for scaling embedding
        self.d_model = d_model
        #  Initialized linear layer
        self.linear_layer = nn.Linear(d_model, target_size)


    def forward(self, target, source, target_mask, source_mask):
        '''
        :param target: outputs of positional embedding / Add & Norm
        :param source: inputs
        :param target_mask: mask
        :param source_mask: mask
        '''
        batch_size, seq_length = target.shape[0], target.shape[1]
       
        position = torch.arange(0, seq_length).repeat(batch_size, 1).to(device)
        output = self.outputEmbedding(target) * math.sqrt(self.d_model)
        output = self.positionalEncoding(output)
        
        for layer in self.layers:
            output, attention = layer(output, source, target_mask, source_mask)

        return self.linear_layer(output), attention

class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=512,
        h=8,
        d_ff=2048,
        P_drop=0.1
        ):
        '''
        :param d_model: dimension of model
        :param h: number of heads in multiheaded attention
        :param d_ff: dimension of the feedforward network
        :param P_drop: dropout rate
        '''
        
        super(DecoderLayer, self).__init__()
        #  Norm after Masked Multi-Head Attention in the decoder block
        self.norm1 = nn.LayerNorm(d_model)
        #  Norm after Multi-head Attention in the decoder block
        self.norm2 = nn.LayerNorm(d_model)
        #  Norm after the feed forward block in the decoder block
        self.norm3 = nn.LayerNorm(d_model)

        
        # self.masked_self_attention = nn.MultiheadAttention(d_model, h, P_drop)
        self.masked_self_attention = MultiHeadAttention(d_model, h, P_drop)
        
        # self.attention = nn.MultiheadAttention(d_model, h, P_drop)
        self.attention = MultiHeadAttention(d_model, h, P_drop)
        
        
        #  Setting up Feed Forward block for decoder
        self.feedForward = FeedForward(d_model, d_ff, P_drop)
        #  Intializing dropout
        self.dropout = nn.Dropout(P_drop)
        
    
    
    def forward(self, target, source, target_mask, source_mask):
        '''
        :param target:
        :param source:
        :param target_mask:
        :param source_mask:
        :return:
        '''
        #  Masked Multi-head Attention. This block is masked self attention so
        #  keys, queries, and values are all target.
        #  We create a temp variable because we have to keep track of the updated
        #  as well as the original target. We have to feed these values over.
        temp, _ = self.masked_self_attention(target, target, target, target_mask)
        
        target = self.norm1(self.dropout(temp) + target)
       
        # temp, attention = self.attention(target.permute(1,0,2), source, source)
        # temp = temp.permute(1,0,2)
        temp, attention = self.attention(source ,source, target)
       
        # temp = temp.permute(1, 0, 2)
        target = self.norm2(self.dropout(temp) + target)

        temp = self.feedForward(target)

        return self.norm3(self.dropout(temp) + target), attention

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model=512,
        h=8,
        P_drop=0.1
        ):
        '''
        :param d_model: dimension of model
        :param h: number of heads for multi-headed attention
        :param P_drop: dropout rate
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.V = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.Q = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(P_drop)


    def forward(self, Value, Key, Query, mask=None):
        '''
        :param Value: V
        :param Key: K
        :param Query: Q
        :param mask: mask
        :return: attention probabilities
        '''
        batch_size = Query.shape[0]
        
        value = self.V(Value)
        key = self.K(Key)
        query = self.Q(Query) 
  
        query = query.view(batch_size, -1, self.h, self.d_model // self.h).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.h, self.d_model // self.h).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.h, self.d_model // self.h).permute(0, 2, 1, 3)
  
        energy = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(self.d_model)
       
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
        x = torch.matmul(self.dropout(attention), value)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        x = self.linear(x)
        return x, attention

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        '''
        :param encoder: initialized object from the Encoder class
        :param decoder: initialized object from the Decoder class
        :param src_pad_idx: padding index of the source (inputs)
        :param trg_pad_idx: padding index of the target (outputs)
        '''
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, source):
        '''
        :param source: inputs
        :return: mask of inputs
        '''
        return (source != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_trg_mask(self, target):
        '''
        :param target: outputs
        :return: mask & sub mask
        ''' 
        target_pad_mask = (target != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        target_len = target.shape[1]
        target_sub_mask = torch.tril(torch.ones((target_len, target_len), device = device)).bool()       
        return target_pad_mask & target_sub_mask
    

    def forward(self, source, target):   
        '''
        :param source: inputs (source)
        :param target: outputs (target)
        :return: Output probabilities and attention from Decoder
        '''     
        source_mask = self.make_src_mask(source)
        target_mask = self.make_trg_mask(target)
        encoder_output = self.encoder(source, source_mask)
        return self.decoder(target, encoder_output, target_mask, source_mask)