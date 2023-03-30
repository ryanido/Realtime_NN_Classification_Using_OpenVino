import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
import yolo_v3_live as y3
import mobilenet_v2_live as mn

textFont = ("Product Sans", 20)
deviceName = 'CPU'

def yolov3 ():
    app.withdraw()
    y3.yolo_v3_live(device_name=deviceName)
    app.show_frame(StartPage)
    app.deiconify()

def mobilenet ():
    app.withdraw()
    mn.mobilenet_v2_live(device_name=deviceName)
    app.show_frame(StartPage)
    app.deiconify()

def on_enter(e):
   e.widget.config(background='#001f3b', foreground= "white")

def on_leave(e):
   e.widget.config(background= '#003c71', foreground= 'white')

def setCPU():
     global deviceName
     deviceName = 'CPU'

def setGPU():
     global deviceName
     deviceName = 'GPU'

LARGEFONT =("Verdana", 35)

class tkinterApp(tk.Tk):
	
	# __init__ function for class tkinterApp
	def __init__(self, *args, **kwargs):
		
		# __init__ function for class Tk
		tk.Tk.__init__(self, *args, **kwargs)
		
		# creating a container
		container = tk.Frame(self)
		container.pack(side = "top", fill = "both", expand = True)

		container.grid_rowconfigure(0, weight = 1)
		container.grid_columnconfigure(0, weight = 1)

		# initializing frames to an empty array
		self.frames = {}

		# iterating through a tuple consisting
		# of the different page layouts
		for F in (StartPage, Page1, Page2 ):

			frame = F(container, self)

			# initializing frame of that object from
			# startpage, page1, page2 respectively with
			# for loop
			self.frames[F] = frame

			frame.grid(row = 0, column = 0, sticky ="nsew")

		self.show_frame(StartPage)

	# to display the current frame passed as
	# parameter
	def show_frame(self, cont):
		frame = self.frames[cont]
		frame.tkraise()

# first window frame startpage

class StartPage(tk.Frame):
   def __init__(self, parent, controller):
      tk.Frame.__init__(self, parent)
		
		# label of frame Layout 2
      logo = Image.open('img/logo.png')
      logo = logo.resize((384, 288), Image.LANCZOS)
      global img
      img = ImageTk.PhotoImage(logo)
      label = ttk.Label(self, image=img)

		# putting the grid in its place by using
		# grid
      label.place(relx=0.5, rely=0.3, anchor='center')

      button = tk.Button(self, text ="Start", fg="white", bg="#003c71", activebackground="#001f3b", activeforeground="white",
                   width=10, height=1, font = textFont, bd =0, command = lambda : controller.show_frame(Page1))

		# putting the button in its place by
		# using grid
      button.place(relx=0.5, rely=0.7, anchor='center')
      button.bind('<Enter>', on_enter)
      button.bind('<Leave>', on_leave)

		


# second window frame page1
class Page1(tk.Frame):
	
   def __init__(self, parent, controller):
		
      tk.Frame.__init__(self, parent)
      logo = Image.open('img/logo.png')
      logo = logo.resize((128, 90), Image.LANCZOS)
      global img2
      img2 = ImageTk.PhotoImage(logo)
      label = ttk.Label(self, image=img2)

		# putting the grid in its place by using
		# grid
      label.place(relx=0.1, rely=0.1, anchor='center')

      # button to show frame 2 with text
      # layout2
      button1 = tk.Button(self, text ="CPU", fg="white", bg="#003c71", activebackground="#001f3b", activeforeground="white",
                   width=10, height=1, font = textFont, bd =0, command = lambda : [setCPU, controller.show_frame(Page2)])

		# putting the button in its place by
		# using grid
      button1.place(relx=0.5, rely=0.4, anchor='center')
      button1.bind('<Enter>', on_enter)
      button1.bind('<Leave>', on_leave)

      button2 = tk.Button(self, text ="GPU", fg="white", bg="#003c71", activebackground="#001f3b", activeforeground="white",
                   width=10, height=1, font = textFont, bd =0, command = lambda : [setGPU, controller.show_frame(Page2)])

		# putting the button in its place by
		# using grid
      button2.place(relx=0.5, rely=0.7, anchor='center')
      button2.bind('<Enter>', on_enter)
      button2.bind('<Leave>', on_leave)




# third window frame page2
class Page2(tk.Frame):
   def __init__(self, parent, controller):
		
      tk.Frame.__init__(self, parent)
      logo = Image.open('img/logo.png')
      logo = logo.resize((128, 90), Image.LANCZOS)
      global img2
      img2 = ImageTk.PhotoImage(logo)
      label = ttk.Label(self, image=img2)

		# putting the grid in its place by using
		# grid
      label.place(relx=0.1, rely=0.1, anchor='center')

      # button to show frame 2 with text
      # layout2
      button1 = tk.Button(self, text ="Yolo-v3", fg="white", bg="#003c71", activebackground="#001f3b", activeforeground="white",
                   width=10, height=1, font = textFont, bd =0, command = yolov3)

		# putting the button in its place by
		# using grid
      button1.place(relx=0.5, rely=0.4, anchor='center')
      button1.bind('<Enter>', on_enter)
      button1.bind('<Leave>', on_leave)

      button2 = tk.Button(self, text ="MobileNet-v2", fg="white", bg="#003c71", activebackground="#001f3b", activeforeground="white",
                   width=10, height=1, font = textFont, bd =0, command = mobilenet)

		# # putting the button in its place by
		# # using grid
      button2.place(relx=0.5, rely=0.7, anchor='center')
      button2.bind('<Enter>', on_enter)
      button2.bind('<Leave>', on_leave)


# Driver Code
app = tkinterApp()
app.title("OpenVINO")
app.geometry("640x360")
app.iconbitmap("img/icon.ico")
app.mainloop()