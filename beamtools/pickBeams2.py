#!/usr/bin/env python
"""
Created on Wed Mar 25 19:53:46 2015

@author: John Swoboda
"""

from Tkinter import *
import numpy as np
import pdb

def rect(r, w, deg=1):
    # radian if deg=0; degree if deg=1
    from math import cos, sin, pi
    if deg:
        w = pi * w / 180.0
    return r * cos(w), r * sin(w)

def polar(x, y, deg=1):
    # radian if deg=0; degree if deg=1
    from math import hypot, atan2, pi
    if deg:
        return hypot(x, y), 180.0 * atan2(y, x) / pi
    else:
        return hypot(x, y), atan2(y, x)


class Gui():
    def __init__(self,root):
        self.root = root
        self.root.title("Beam Selector")

        self.output = []
        self.beamhandle = None
        self.sizecanv = [800,800]
        self.off_x = self.sizecanv[0]/2
        self.off_y = self.sizecanv[1]/2
        self.div = 60
        self.lat = [80,70,60,50,40,30]
        self.angles = np.arange(0,180,30)
        self.frame = Frame(self.root,background="gray")
        self.frame.grid(row=0,column=0, sticky="n")
        # set up the canvas
        self.canv = Canvas(self.frame , width=self.sizecanv[0], height=self.sizecanv[1],background='white')
        self.canv.grid(row=1,column=0)
        self.canv.pack(fill="both", expand=True)


        self.var = StringVar()
        self.var.set("PFISR")
        self.choices = {"PFISR":'PRISRbeammap.txt', "RISR-N":'RISRNbeammap.txt'}#, "RISR-S":'file3'}
        self.option = OptionMenu(self.frame, self.var, *self.choices)
        self.option.grid(row=0,column=0,sticky="nw")
        self.option.pack()
        self.lines = np.loadtxt(self.choices[self.var.get()])
        self.Drawlines()
        self.Drawbeams()

        self.butt = Button(self.frame, text="Finished", command=self.buttonClick)
        self.butt.grid(row=0,column=1,sticky='ne')
        self.butt.pack()
        self.readfile = StringVar()
        self.canv.bind('<ButtonPress-1>', self.onCanvasClick)
        self.canv.bind('<ButtonPress-2>', self.onCanvasRightClick)
        self.var.trace('w', self.Changefile)


    def Changefile(self,*args):
        """ This function will change the files to a different radar system."""
        filename= self.choices[self.var.get()]
        self.readfile.set(filename)
        self.lines = np.loadtxt(filename)
        self.output=[]
        self.Drawbeams()

    def Drawbeams(self):
        "This function will "
        div =self.div
        if self.beamhandle is not None:
            self.canv.delete(ALL)
            self.Drawlines()
        off_x = self.off_x
        off_y = self.off_y
        for beams in self.lines:
            c_coords = rect(90-beams[2],beams[1]-90)
            #print c_coords
            points = [c_coords[0]*div/10+off_x -5,
                    c_coords[1]*div/10+off_y-5,
                    c_coords[0]*div/10+off_x +5,
                    c_coords[1]*div/10+off_y+5]
            self.beamhandle = self.canv.create_oval(points, fill='blue',tags='beams')

    def Drawlines(self):

        off_x = self.off_x
        off_y = self.off_y
        div = self.div
        lat = self.lat
        Nlat = len(lat)
        angles = self.angles
        # Create circles

        textangles = {iang:(str(np.mod(270+iang,360)),str(np.mod(90+iang,360))) for iang in angles}

        for i in range(1,7):
            points = [-div*i+off_x, -div*i+off_y, div*i+off_x, div*i+off_y]
            self.canv.create_oval(points,dash=(5,5))
            self.canv.create_text(points[0]+(div*i)/(div/15),points[1]+(div*i)/(div/15),text=str(lat[i-1]))
        for iang in angles:
            iangr = iang*np.pi/180
            cosan = np.cos(iangr)
            sinan = np.sin(iangr)

            points = [-div*Nlat*cosan+off_x, -div*Nlat*sinan+off_y, div*Nlat*cosan+off_x, div*Nlat*sinan+off_y]
            self.canv.create_line(points,dash=(5,5))
            self.canv.create_text(points[0],points[1], text=textangles[iang][0])
            self.canv.create_text(points[2],points[3], text=textangles[iang][1])

    def onCanvasClick(self,event):
        if event.widget.find_closest(event.x, event.y) == self.canv.find_withtag('finish'):
            print "Need to fix this"
        else:
            p_coords = polar(event.x-500,event.y-500)
            print  'Clicked Az: ', p_coords[1],'El: ', np.abs(90 - p_coords[0]/6)
            closest = self.lines[np.int32(event.widget.find_closest(event.x, event.y))-1]
            textheader = 'Closest beam is # %s, Az: %s, El: %s' %(closest[0][0],closest[0][1],closest[0][2])
            self.canv.itemconfig(self.canv.find_withtag('header'),text=textheader)
            self.canv.itemconfig(event.widget.find_closest(event.x, event.y), fill='orange')
            print 'Closest beam is # %s, Az: %s, El: %s' %(closest[0][0],closest[0][1],closest[0][2])
            self.output.append(closest[0][0])
    def buttonClick(self):
        print 'Finished'
        print self.output
        f = open('SelectedBeamCodes.txt', 'w')
        for beam in self.output:
            f.write("%s\n" % (beam))
        sys.exit()

    def onCanvasRightClick(self,event):
        if np.int32(event.widget.find_closest(event.x, event.y)) < 481:
            self.canv.itemconfig(event.widget.find_closest(event.x, event.y), fill='blue')
            closest = self.lines[np.int32(event.widget.find_closest(event.x, event.y))-1]
            if closest[0][0] in self.output:
                self.output.remove(closest[0][0])
if __name__ == "__main__":

    root = Tk()
    gui = Gui(root)
    root.mainloop()