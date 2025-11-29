import tkinter as tk

from config import ConfiguracionGlobal
from gui import StereoAppTkinter

if __name__ == '__main__':
    root = tk.Tk()
    config = ConfiguracionGlobal(nom_vid="dummy.mp4")
    app = StereoAppTkinter(root, config)
    root.mainloop()
