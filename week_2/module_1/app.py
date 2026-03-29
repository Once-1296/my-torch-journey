import tkinter as tk
from PIL import Image, ImageDraw,ImageOps

class_to_cloth = {

    0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit/Fashion Drawer")
        
        # 1. Setup Canvas
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        
        # 2. Setup "Internal" PIL image to save later
        # We draw on both the UI and this hidden image
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn = tk.Button(root, text="Classify", command=self.classify)
        btn.pack(side= tk.LEFT)
        clear_btn = tk.Button(root, text="Clear screen", command=self.clear)
        clear_btn.pack(side=tk.RIGHT)

        from model import model as fashion_model
        self.fashion_model = fashion_model
        from model import train_call
        self.fashion_model = train_call(model=self.fashion_model)
        


    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)
        self.draw.line([x1, y1, x2, y2], fill="black", width=10)

    def classify(self):
        # 1. Resize and Invert
        # FashionMNIST is white drawings on black background. 
        # Since you draw black on white, we MUST invert it.
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img) 
        
        # 2. Convert to Tensor
        # We create the transform "tool" and then apply it\
        from torchvision import transforms
        transform_tool = transforms.ToTensor()
        img_tensor = transform_tool(img) # Shape: [1, 28, 28]
        
        # 3. Add the Batch Dimension
        # Model expects [1, 1, 28, 28]. unsqueeze(0) adds a '1' at the start.
        img_tensor = img_tensor.unsqueeze(0) 

        # 4. Inference
        self.fashion_model.eval() # Set to evaluation mode!
        import torch
        with torch.no_grad(): # Don't calculate gradients during prediction
            output = self.fashion_model(img_tensor)
            pred_idx = output.argmax(dim=1).item() # .item() turns it into a Python int
        
        # 5. Show Result
        cloth_name = class_to_cloth.get(pred_idx, "Unknown")
        from tkinter import messagebox
        messagebox.showinfo("Prediction", f"I think this is a: {cloth_name}")


    def clear(self):
        self.canvas.create_rectangle(0, 0, 280, 280, fill="white")
        self.draw.rectangle([0, 0, 280, 280], fill="white")

if __name__ == "__main__":
    root = tk.Tk()
    App = DrawingApp(root)
    import sys

    def on_closing():
        root.destroy()
        sys.exit()  # Forces the process to terminate
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()