import tkinter as tk
from PIL import Image, ImageDraw,ImageOps
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



class_to_cloth = {

    0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1400x850")
        self.root.title("Digit/Fashion Drawer")
        
        
        #0. Set up frames
        self.sidebar = tk.Frame(self.root, width=200,height=800)
        self.sidebar.pack(side=tk.LEFT,padx=10,pady=10)
        self.paintframe = tk.Frame(self.root,width=1000,height=800)
        self.paintframe.pack(side=tk.RIGHT)
        self.resFrame = tk.Frame(self.sidebar)
        self.resFrame.pack()
        
        # 1. Setup Canvas
        self.canvas = tk.Canvas(self.paintframe, width=1000, height=800, bg="white")
        self.canvas.pack()
        
        # 2. Setup "Internal" PIL image to save later
        # We draw on both the UI and this hidden image
        self.image = Image.new("L", (1000, 800), "white")
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        btn = tk.Button(self.sidebar, text="Classify", command=self.classify)
        btn.pack(side= tk.TOP)
        clear_btn = tk.Button(self.sidebar, text="Clear screen", command=self.clear)
        clear_btn.pack(side=tk.TOP)
        
        # 3 result graph
        # 3.1. Create a Matplotlib Figure and Axis
        # figsize is in inches (width, height)
        self.fig, self.ax = plt.subplots(figsize=(4, 3)) 

        # 3.2. Link the Figure to your Tkinter Frame (e.g., self.resFrame)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.resFrame)

        # 3.3. Pack the canvas widget into the UI
        self.canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 4. Model Initialization (Do this ONCE at startup)
        from model import DeepFashionClassifier
        from constants import MODEL_FILE
        import torch
        import os

        self.fashion_model = DeepFashionClassifier()
        
        if not os.path.exists(MODEL_FILE):
            print("Model not found. Training now...")
            from train_and_save import main
            main()
            
        # Load the weights into the model structure
        self.fashion_model.load_state_dict(
            torch.load(MODEL_FILE, map_location=torch.device('cpu'))
        )
        self.fashion_model.eval() 
        print("Model loaded and ready!")
        


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
        import torch
        with torch.no_grad(): # Don't calculate gradients during prediction
            output = self.fashion_model(img_tensor)
            import torch.nn.functional as F
            probs = F.softmax(output,dim=1)
            top3 = torch.topk(probs,3)
            # print(top3.indices.tolist(),type(top3.indices.tolist()),sep='\n')
            self.lis,self.plis = top3.indices.tolist()[0],top3.values.tolist()[0]
        
        # 5. Show Result
        self.plot_result()

    def plot_result(self):
        # Clear the old bar chart
        self.ax.clear()

        # Draw the new one (x_labels, y_values)
        self.ax.bar([class_to_cloth[x] for x in self.lis], self.plis, color=['blue' if x % 3==0 else 'orange' if x %3==1 else 'green' for x in range(len(self.lis))])

        # Set the Y-axis from 0.0 to 1.0 (since probabilities max out at 1.0)
        self.ax.set_ylim(0, 1)

        # Push the updates to the Tkinter UI
        self.canvas_plot.draw()
        
    def clear(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 1000, 800, fill="white")
        self.draw.rectangle([0, 0, 1000, 800], fill="white")

if __name__ == "__main__":
    root = tk.Tk()
    App = DrawingApp(root)
    import sys

    def on_closing():
        root.destroy()
        sys.exit()  # Forces the process to terminate
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()