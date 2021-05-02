import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import pandas as pd
from jiwer import wer
from variables_ecog import ENCODER_INPUT, DECODER_INPUT, device
import pdb

class Validate:
    
    def __init__(self, data, vocab, max_iter=100, att_list=None, df=None, WER=None):
        self.data = data
        self.vocab = {v: k for k, v in vocab.items()}
        self.att_list = att_list
        self.df = df
        self.WER = WER
        self.min_WER = 1.0
        self.min_df = df
        self.max_iter = max_iter


    def index_to_text(self, input):
        output = list()
        for i in input:
            if i < len(self.vocab) and i >= 0:
                output.append(self.vocab[i][:-1])
            else:
                output.append(self.vocab[2][:-1])
        return output
    

    def display_attention(self, index, n_heads = 8, n_rows = 4, n_cols = 2):
    
        assert n_rows * n_cols == n_heads
        
        attention = self.att_list[index]
        
        fig = plt.figure(figsize=(25,35))
        
        for i in range(n_heads):
            
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            
            _attention = attention.squeeze(0)[i].cpu().detach().numpy()

            cax = ax.matshow(_attention, cmap='bone')
    
            ax.tick_params(labelsize=12)

        plt.show()
        plt.close()


    def translate_sentence(self, sentence, model, device, max_len = 100, bayes=False):
        model.eval()
        trg_indexes = list()
        src_tensor = torch.tensor([sentence]).to(device)
        src_mask = model.make_src_mask(src_tensor)
        
        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_mask)
        
        trg_indexes.append(0)
        
        for i in range(max_len - 1):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

            trg_mask = model.make_trg_mask(trg_tensor)
        
            with torch.no_grad():
          
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            
          
            pred_token = output.argmax(2)[:,-1].item()
        
            trg_indexes.append(pred_token)
            
            if pred_token == 1:
                break
            

        return trg_indexes, attention

    def create_df(self, real, hypothesis):
        d = {'Actual':real, 'Predicted':hypothesis}
        return pd.DataFrame(data=d)
        
    def validate(self, model):
        
        hypothesis = list()
        real = list()
        self.att_list = list()
        
        for i, d in enumerate(self.data):
            src = d[ENCODER_INPUT]
            EOS_INDEX = d[DECODER_INPUT].index(1)
            max_length = len(d[DECODER_INPUT][:EOS_INDEX + 1])
            temp = d[DECODER_INPUT]
            real.append(' '.join(self.index_to_text(temp)[1:EOS_INDEX]))
            translation, attention = self.translate_sentence(src, model, device, max_length)
            self.att_list.append(attention)
            hypothesis.append(' '.join(self.index_to_text(translation)[1:-1]))
            
        self.df = self.create_df(real, hypothesis)
        self.WER = 0
        for i in range(len(real)):
            WER = wer(real[i], hypothesis[i])
            self.WER += WER
        self.WER /= len(real)
        
        
        if self.WER < self.min_WER:
            self.min_WER = self.WER
            self.min_df = self.df.copy()
        
        return self.WER