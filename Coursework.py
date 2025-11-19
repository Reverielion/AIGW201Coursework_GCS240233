import lmstudio as lms
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import function_library as fl
import pandas as pd
from PredictForest import PredictForest
from PCAGMMClustering import PCAGMM
from NaiveBayesClass import NaiveBayesClass
from PIL import ImageTk, Image

class FairyTheSupremeAssistant:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("AI Assistant")
        self.window.geometry("1920x1080")
        self.window.state("zoomed")
        self.window.configure(bg="#27272A")
        self.modelquickbutdumb = lms.llm("llama-3.2-1b-instruct")
        self.modelsmartbutslow = lms.llm("openai/gpt-oss-20b")
        self.chat = lms.Chat("You are a task focused AI assistant working for a manager, if asked to open zalo or notepad, say Sure Thing!")
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_rowconfigure(2, weight=1)
        self.window.grid_columnconfigure(0, weight=7)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_columnconfigure(2, weight=1)
        self.window.grid_columnconfigure(3, weight=1)
        self.window.grid_columnconfigure(4, weight=1)
        self.prompt = ""
        self.predator = PredictForest()
        self.naivebayes = None

        self.create_layout()
        self.window.mainloop()

    def create_layout(self):

        self.output = scrolledtext.ScrolledText(self.window, width=100, height=20)
        self.output.grid(row=0, column=0)
        self.output.configure(state="disabled")

        self.prompt_entry = tk.Text(self.window, width=100, height=5)
        self.prompt_entry.grid(row=2,column=0)

        self.submitbutton = tk.Button(self.window, height=2, width=20, text="Submit", command=self.aichan)
        self.submitbutton.grid(row=2, column=1)

        self.othersubmitbutton = tk.Button(self.window, height=2, width=20, text="Data Analysis Submit", command=self.data_analysis_path)
        self.othersubmitbutton.grid(row=1, column=1)

        self.predicttrain = tk.Button(self.window, height=2, width=20, text="Train on dataset(Numerical)", command=self.predict_train)
        self.predicttrain.grid(row=1, column=2)

        self.predictinput = tk.Button(self.window, height=2, width=20, text="Predict(Numerical)", command=self.predictmanual)
        self.predictinput.grid(row=1, column=3)

        self.clusteringbutton = tk.Button(self.window, height=2, width=20, text="Clustering", command=self.clusterringset)
        self.clusteringbutton.grid(row=2, column=4)

        self.clusteringbutton = tk.Button(self.window, height=2, width=20, text="Train on dataset(Categorical)", command=self.categoricaltrain)
        self.clusteringbutton.grid(row=2, column=2)

        self.clusteringbutton = tk.Button(self.window, height=2, width=20, text="Predict(Categorical)", command=self.categoricalpredict)
        self.clusteringbutton.grid(row=2, column=3)

        self.image_slot = tk.Label(self.window, bg="#27272A")
        self.image_slot.grid(row=0, column=1, columnspan=2)

    def categoricalpredict(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        result, prolly = self.naivebayes.predict(prompt)
        self.output.config(state="normal")
        self.output.insert(tk.END, str(result))
        self.output.insert(tk.END, f"\nProbabilities:\n{str(prolly)}\n")
        self.output.configure(state="disabled")

    def categoricaltrain(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        self.naivebayes = NaiveBayesClass(prompt)
        self.output.config(state="normal")
        self.output.insert(tk.END, "Training completed. Ready to predict\n")
        self.output.configure(state="disabled")

    def predictmanual(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        self.predictresult = self.predator.manual_predict_input(prompt)
        self.output.config(state="normal")
        self.output.insert(tk.END, str(self.predictresult))
        self.output.insert(tk.END, "\n")
        self.output.configure(state="disabled")

    def predict_train(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        self.predator.retrain(prompt)
        self.output.config(state="normal")
        self.output.insert(tk.END, "Training completed. Please check predictiontest.png.\n")
        self.output.configure(state="disabled")
        self.predator.imageplot()
        self.img = Image.open("predictiontest.png")
        self.img = self.img.resize((300, 300), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(self.img)
        self.image_slot.config(image=self.img)

    def clusterringset(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        self.principal = PCAGMM(prompt)
        self.img = Image.open("bestplot.png")
        self.img = self.img.resize((300, 300), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(self.img)
        self.image_slot.config(image=self.img)
        self.image_slot.image = self.img
        self.output.config(state="normal")
        self.output.insert(tk.END, "Clustering completed. Please check bestplot.png.\n")
        self.output.configure(state="disabled")

    def data_analysis_path(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        print(prompt)
        with open(prompt, "r") as file:
            content = file.read()
        self.chat.add_user_message("Perform general data analysis: What it is, Average, Highest and lowest Score for each columm")
        result = self.modelquickbutdumb.respond(content)
        self.output.config(state="normal")
        self.output.insert(tk.END, result)
        self.output.insert(tk.END, "\n")
        self.output.configure(state="disabled")
        print(result)

    def aichan(self):
        prompt = self.prompt_entry.get("1.0", tk.END).strip()
        print(prompt)
        self.chat.add_user_message(prompt)
        self.toolshed = [fl.open_Notepad, fl.open_Zalo]
        act_result = self.modelsmartbutslow.act(
            chat=self.chat,
            tools=self.toolshed,
        )
        result = self.modelsmartbutslow.respond(self.chat, on_message=self.chat.append)
        answer = result.content
        x = answer.split("<|message|>")
        x[2] = (f"User: {prompt}\n"
                f"AI Assistant: {x[2]}\n\n")
        self.output.config(state="normal")
        self.output.insert(tk.END, x[2])
        self.output.configure(state="disabled")

if __name__ == "__main__":
    FairyTheSupremeAssistant()