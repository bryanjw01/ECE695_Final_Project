import torch
from variables_ecog import ENCODER_INPUT, DECODER_INPUT, PHONEME_INPUT, device
import pdb
from variables import DEVICE

class Train:

    def __init__(self, data, fake, fake_sentences):
        self.iterator = data
        self.fake_iterator = fake
        self.fake_sentences = fake_sentences

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
            src = torch.tensor([batch[0]]).to(device)
            trg = torch.tensor([batch[1]]).to(device)
            src = src[None, :]
            pdb.set_trace()
            optimizer.zero_grad()
            output, _ = model(self.src, self.trg[:,:-1]) 
            output_dim = output.shape[-1]
                
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / (max_length + 1), optimizer, criterion

    def train_fake(self, model, optimizer, criterion, clip):
        model.train()
        
        epoch_loss = 0
        len_iterator = 0
        max_length = 0

        for i, batch in enumerate(self.iterator):
            max_length = i
            src = torch.tensor([batch[0]]).to(device)
            trg = torch.tensor([batch[1]]).to(device)
            
            optimizer.zero_grad()
            output, _ = model(src, trg[:,:-1]) 
            output_dim = output.shape[-1]   
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
        src = torch.tensor(self.fake_iterator).to(DEVICE)
        trg = torch.tensor(self.fake_sentences).to(DEVICE)

        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1]) 
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg) * 0.01
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        return epoch_loss / (max_length + 2), optimizer, criterion

class TrainGAN:
    def __init__(self, gen_opt, dis_opt, criterion):
        self.gen_opt = gen_opt
        self.dis_opt = dis_opt
        self.criterion = criterion
        self.noise = torch.randn(64, 100, 1, 1, device=device)
        self.label = [0., 1.]
        self.gen_loss = list()
        self.dis_loss = list()
        self.images = list()
        

    def train(self, gen_model, dis_model, data, epoch):
        for i, data in enumerate(data, 0):
            dis_model.zero_grad()
            input = data[0].to(device=DEVICE)
            batch_size = input.size(0)
            label = torch.full((batch_size,), self.label[1], dtype=torch.float, device=DEVICE)
            output = dis_model(input).view(-1)
            dis_loss = self.criterion(output, label)
            dis_loss.backward()
            noise = torch.randn(batch_size, 100, 1, 1, device=DEVICE)
            fake = gen_model(noise)
            label.fill_(self.label[0])
            output = dis_model(fake.detach()).view(-1)
            dis_loss = self.criterion(output, label) 
            dis_loss.backward()
            self.dis_opt.step()
            gen_model.zero_grad()
            label.fill_(self.label[1])  
            output = dis_model(fake).view(-1)
            gen_loss = self.criterion(output, label)
            gen_loss.backward()
            self.gen_opt.step()
            self.gen_loss.append(gen_loss.item())
            self.dis_loss.append(dis_loss.item())

    def train_ecog(self, gen_model, dis_model, data, epoch):
        
        for i in range(8):
            dis_model.zero_grad()
            input = torch.tensor(data.dataset[50*i:50*(i+1),0].tolist()).to(device=DEVICE)
            batch_size = input.size(0)
            label = torch.full((batch_size,), self.label[1], dtype=torch.float, device=DEVICE)
            output = dis_model(input).view(-1)
            dis_loss = self.criterion(output, label)
            dis_loss.backward()
            noise = torch.randn(batch_size, 100, 1, device=DEVICE)
            fake = gen_model(noise)
            label.fill_(self.label[0])
            output = dis_model(fake.detach()).view(-1)
            dis_loss = self.criterion(output, label) 
            dis_loss.backward()
            self.dis_opt.step()
            gen_model.zero_grad()
            label.fill_(self.label[1])  
            output = dis_model(fake).view(-1)
            gen_loss = self.criterion(output, label)
            gen_loss.backward()
            self.gen_opt.step()
            self.gen_loss.append(gen_loss.item())
            self.dis_loss.append(dis_loss.item())
    

    
    
    
    
