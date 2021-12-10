# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:45:36 2021

@author: Riccardo Iacobucci
"""
import warnings
import numpy as np

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')

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

def movelr(board): #this refers to only two direction (left-right), can be called two times for the 4 moves 
    
    helpermat = np.array([[1,2,3],[1,2,3],[1,2,3],[1,2,3]]);
    
    # moving
    #mover = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]];
    b = np.sum(board,0);
    # movinglines = np.matmul(b,mover);
    # movablelines = (np.max(movinglines,1)==1);
    
    optmovesmat=b[:,0:3]-b[:,1:4];
    optmovesL=optmovesmat<0;
    optmovesR=optmovesmat>0;
    
    boardnum = decodeboard(board);
    optpowersmat = decodeboard(board,np.arange(2,15));
    optthreesmat = decodeboard(board,np.arange(0,2));
    
    optpowersmat = np.where(optpowersmat==0,np.nan,optpowersmat);
    optpowers = (optpowersmat[:,0:3]-optpowersmat[:,1:4]==0);
    
    optthrees=optthreesmat[:,0:3]+optthreesmat[:,1:4]==3;
    
    optcombiL = (optthrees+optpowers+optmovesL)*helpermat;
    optcombiR = (optthrees+optpowers+optmovesR)*helpermat;
    
    optcombiL = np.where(optcombiL==0,np.nan,optcombiL);
    optcombiR = np.where(optcombiR==0,np.nan,optcombiR);
    
    activelinesL = np.nansum(optcombiL,1)>0;
    activelinesR = np.nansum(optcombiR,1)>0;
    
    movementL = np.nanmin(optcombiL,1).astype(int)-1;
    movementR = np.nanmax(optcombiR,1).astype(int)-1;
    
    newbL = np.copy(boardnum);
    newbR = np.copy(boardnum);
    
    for i in range(0,4):
        if activelinesL[i]:
            newbL[i,movementL[i]:3] = boardnum[i,movementL[i]+1:];
            newbL[i,movementL[i]] = boardnum[i,movementL[i]] + boardnum[i,movementL[i]+1];
            newbL[i,3]=0;
        if activelinesR[i]:
            newbR[i,1:movementR[i]+1] = boardnum[i,0:movementR[i]];
            newbR[i,movementR[i]+1] = boardnum[i,movementR[i]] + boardnum[i,movementR[i]+1];
            newbR[i,0]=0;
            
    return (activelinesL,activelinesR,newbL,newbR); 


class tre:
    
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
            
            if move==0:
                 newb[move,nextpiece_pos,3] = self.nextpiece;
            if move==1:
                 newb[move,nextpiece_pos,0] = self.nextpiece;
            if move==2:
                 newb[move,3,nextpiece_pos] = self.nextpiece;
            if move==3:
                 newb[move,0,nextpiece_pos] = self.nextpiece;
            
            self.board = encodeboard(newb[move,:,:]);
            if self.simplified==True:
                self.nextpiece = 3;
            else:
                self.nextpiece = np.random.randint(1,4);
            
            
        return (moved,done)
    
    def possiblemoves(self): 
        
        activelinesL,activelinesR,newbL,newbR = movelr(self.board);
        activelinesU,activelinesD,newbU,newbD = movelr(np.transpose(self.board,(0,2,1)));
        
        newbU=np.transpose(newbU);
        newbD=np.transpose(newbD);
        
        activelines=np.stack((activelinesL,activelinesR,activelinesU,activelinesD));
        newb=np.stack((newbL,newbR,newbU,newbD));
        
        return (activelines,newb);
        
    def decode(self,rang=np.arange(15)):
        outboard = decodeboard(self.board,rang);
        return outboard.astype(int);
        
    def show(self):
        outboard = self.decode();
        print(outboard);
    

