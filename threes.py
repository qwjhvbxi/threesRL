# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:45:36 2021

@author: Riccardo Iacobucci
"""

import numpy as np


# one-hot encoding order: 
# 1 2 3 6 12 24 48 96 192 384 768 1500 3000 6000 12000 -> 15
# 4x4x15 planes

def encodeboard(board):
    board0 = np.zeros([15,4,4],dtype=int);
    def_numbers = np.concatenate(([1,2],np.exp2(np.arange(13))*3));
    for i in range(0,def_numbers.size):
        board0[i,:,:]=(board==def_numbers[i]).astype(int);
    return board0;
    
def decodeboard(board,rang=np.arange(15)):
    def_numbers = np.concatenate(([1,2],np.exp2(np.arange(13))*3));
    outboard = np.matmul(np.moveaxis(board[rang,:,:],0,2),np.transpose(def_numbers[rang]));
    return outboard.astype(int);

def moveleft(board):
    
    activelines = np.ones((4, 3), dtype=bool);
    
    for i in [0,1,2,3]:
        
        for k in [0,1,2]:
        
            # check if 1 2
            new3 = (board[0,i,k]^board[0,i,k+1]) & (board[1,i,k]^board[1,i,k+1]);
            
            if new3:
                board[0:2,i,k:k+2] = 0;
                board[2,i,k] = 1;
            else:
                
                # check if any same number
                newup = board[2:,i,k] & board[2:,i,k+1];
                
                if np.sum(newup)>0: 
                    board[2,i,k] = 0;
                    board[3:,i,k] = newup[0:12];
                    board[2:,i,k+1] = 0;
                else:
                    
                    # check if movement
                    mov = np.any(board[:,i,k]);
                    mov2 = np.any(board[:,i,k+1]);
                    
                    if mov2 and not mov:
                        board[:,i,k] = board[:,i,k+1];
                        board[:,i,k+1] = 0;
                    else:
                        activelines[i,k] = False;
                        
    return (np.any(activelines,1),board);



class tre:
    
    scoreboard = np.append([0,0],np.power(3,np.arange(1,14)));
    def_numbers = np.concatenate(([1,2],np.exp2(np.arange(13))*3));
    
    def __init__(self,simplified=False):
        self.simplified = simplified;
        self.reset();
        
    def maxnum(self):
        maxnum_index = np.where(np.sum(self.board,(1,2)))[0].max();
        return self.def_numbers[maxnum_index];
        
    def state(self):
        nextpiecevec = np.zeros((10,1));
        nextpiecevec[self.nextpiece-1] = 1;
        return np.append(self.board.flatten(),nextpiecevec);
    
    def reset(self):
        if self.simplified==True:
            self.nextpiece = 3;
            k = np.random.choice((-1,2),size=(4,4),p=(0.5,0.5));
        else:
            self.nextpiece = np.random.randint(1,4);
            k = np.random.choice((-1,0,1,2),size=(4,4),p=(0.5,0.2,0.2,0.1));
            
        board = np.zeros([15,4,4],dtype=int);
        
        for i in range(0,4):
            for j in range(0,4):
                if k[i,j]>=0:
                    board[k[i,j],i,j]=1;
                    
        self.board = board;
        self.score = np.sum(self.scoreboard*np.sum(np.sum(self.board[:,:,:],1),1));
        return self.state();
    
    def step(self,action):
        moved,done = self.nextmove(action);
        # reward = int(moved)*1.1 - 0.1;
        reward = int(moved)*2 - 1; # negative reward for illegal moves
        return (self.state(),reward,done);
    
    def gioca(self,move):
        moved,done = self.nextmove(move);
        self.show();
    
    def nextmove(self,move): # 0,1,2,3 = L,R,U,D
        
        activelines,newb = self.possiblemoves();
        
        numactivelinestotal = np.sum(activelines);
        numactivelines = np.sum(activelines[move]);
        done = False;
        moved = True;
        
        if numactivelinestotal==0: # lost
            done = True;
            moved = False;
        elif numactivelines==0: # not lost, but move is not allowed
            moved = False;
        else:
            nextpiece_pos = np.random.choice(4,1,p=activelines[move]/numactivelines);
            
            # WARNING! Temporary solution: only works with nextpiece=1,2,3 
            if move==0:
                 newb[move,self.nextpiece-1,nextpiece_pos,3] = 1;
            if move==1:
                 newb[move,self.nextpiece-1,nextpiece_pos,0] = 1;
            if move==2:
                 newb[move,self.nextpiece-1,3,nextpiece_pos] = 1;
            if move==3:
                 newb[move,self.nextpiece-1,0,nextpiece_pos] = 1;
            
            self.board = newb[move,:,:,:];
            if self.simplified==True:
                self.nextpiece = 3;
            else:
                self.nextpiece = np.random.randint(1,4);
            
            self.score = self.scoreboard*np.sum(np.sum(self.board[:,:,:],1),1)
            
        return (moved,done)
    
    def possiblemoves(self):
        
        activelinesL,newbL = moveleft(np.copy(self.board));
        activelinesR,newbR = moveleft(np.flip(np.copy(self.board),2));
        activelinesU,newbU = moveleft(np.rot90(np.copy(self.board),1,(1,2)));
        activelinesD,newbD = moveleft(np.rot90(np.copy(self.board),1,(2,1)));
        
        newbR = np.flip(newbR,2);
        newbU = np.rot90(newbU,1,(2,1));
        newbD = np.rot90(newbD,1,(1,2));
        
        activelines = np.stack((activelinesL,activelinesR,np.flip(activelinesU),activelinesD));
        newb = np.stack((newbL,newbR,newbU,newbD));
        
        return (activelines,newb);
        
    def decode(self,rang=np.arange(15)):
        outboard = decodeboard(self.board,rang);
        return outboard.astype(int);
        
    def show(self):
        outboard = self.decode();
        print(outboard);
    
    
    





