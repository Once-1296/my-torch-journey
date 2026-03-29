from constants import ITER, MODEL_FILE

from model import train_call,model


import torch
def main():
    tmodel = train_call(n=ITER, model=model)
    # SAVE THE STATE DICT, NOT THE MODEL
    torch.save(tmodel.state_dict(), MODEL_FILE) 
    print(f"Model weights saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
