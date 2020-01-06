#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 09:09:46 2019
@author: Pedro, Rodrigo, Alix
"""
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
import os
#Create Window Object

window =Tk()
window.title('Genre Classifier')
window.geometry('600x125')
img = Image.open("note.png")
newsize=(100,100)
img = img.resize(newsize)
image = ImageTk.PhotoImage(img)
panel = Label(window,image=image)
panel.grid(row=0,column=2, columnspan =5, rowspan = 5)

#Define Button Funcions

def close_window ():
    window.destroy()

def open_file():
    global songname
    file = askopenfilename(initialdir = '/home/rodrigo/Music',filetypes = [('Song Files','*.wav *.au *.mp3')])
    if file != '':
        songname = file
        name = songname.split('/')[-1]
        l1.config(text='Name: '+name)

def classify_song(songname):
    if songname is not None:
        #classification = CODE(songname)
        #PLACE CODE TO CLASSIFY HERE!!!
        l2.config(text = 'Genre: ')

btn_choose = Button(window, text="Choose song to classify",width=20, command = lambda:open_file())
btn_choose.grid(row=1,sticky=W)
# btn.grid(row=0,column=1)
try: songname
except NameError: songname = None

l1 = Label(window,text= 'Name: No song selected',width=40)
l1.grid(row=1,column=1)

btn_clas = Button(window, text='Classify',width=20, command = lambda:classify_song(songname))
btn_clas.grid(row=3,sticky=W)
#
l2 = Label(window,text= 'Genre: None',width=20)
l2.grid(row=3,column=1)

btn_quit = Button (window, text = "Quit Classifier.", command = close_window)
btn_quit.grid(row=5,column=1)

window.mainloop()
