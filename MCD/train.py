import torch
from variables import ENCODER_INPUT, DECODER_INPUT, PHONEME_INPUT, device
import pdb

class Train:

    def __init__(self, data):
        self.iterator = data

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def train(self, model, optimizer, criterion, clip):
        model.train()
        
        epoch_loss = 0
        len_iterator = 0
        max_length = 0

        for i, batch in enumerate(self.iterator):
            max_length = i
            src = torch.tensor([batch[ENCODER_INPUT]]).to(device)
            trg = torch.tensor([batch[DECODER_INPUT]]).to(device)
            #FIXME
            trg_ecog = torch.tensor([batch[PHONEME_INPUT]]).to(device)
            # END
            optimizer.zero_grad()
            # FIXME
            output, output_ecog = model(src, trg[:,:-1]) 
            # END
            output_dim = output.shape[-1]
                
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
           
            loss = criterion(output, trg)
            #FIXME
            output_dim = output_ecog.shape[-1]
                
            output = output_ecog.contiguous().view(-1, output_dim)
            trg = trg_ecog[:,:].contiguous().view(-1)
          
            loss += criterion(output, trg) * 0.25
            #END
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / (max_length + 1), optimizer, criterion