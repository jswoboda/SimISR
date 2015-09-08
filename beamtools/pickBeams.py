#!/usr/bin/env python
"""
This GUI is based off of a GUI originally developed by Steven Chen at SRI.
The code was cleaned up so that the GUI is now encompassed in a class structure. Also
the ability to switch between PFISR and RISR-N has been added along with a finish button.
The code also outputs a picture of the selected beam pattern.

@author: John Swoboda
Updated by Greg Starr so it can be used as part of a larger GUI
"""

from Tkinter import *
import os, inspect
import numpy as np
import matplotlib.pyplot as plt
from beamfuncs import BeamSelector

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
    def __init__(self,parent,subgui=True):
         # get the current path
        curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        # set the root
        self.parent = parent
        # set up frames for list
        self.frame1 = Frame(self.parent)
        self.frame1.grid(row=0,column=0)
        self.frame2 = Frame(self.parent)
        self.frame2.grid(row=0,column=1)

        self.output = []
        self.beamhandle = None
        if subgui:
            self.sizecanv = [500,500]
        else:
            self.sizecanv = [1000,1000]
            self.butt = Button(self.frame1, text="Finished", command=self.buttonClick)
            self.butt.grid(row=1,column=1,sticky='w')
            self.leb = Label(self.frame1, text="Beam Selector",font=("Helvetica", 16))
            self.leb.grid(row=0, sticky=W+E+N+S,columnspan=2)


        self.off_x = self.sizecanv[0]/2
        self.off_y = self.sizecanv[1]/2
        self.div = 75.0*self.sizecanv[0]/1000.0
        self.lat = [80,70,60,50,40,30]
        self.angles = np.arange(0,180,30)

        self.var = StringVar()
        self.var.set("PFISR")
        self.choices = {"PFISR":os.path.join(curpath,'PFISRbeammap.txt'), "RISR-N":os.path.join(curpath,'RISRNbeammap.txt')}#, "RISR-S":'file3'}
        self.option = OptionMenu(self.frame1, self.var, *self.choices)
        self.option.grid(row=1,column=0,sticky='w')
        self.lines = np.loadtxt(self.choices[self.var.get()])


        self.readfile = StringVar()

        # set up the canvas
        self.canv = Canvas(self.frame1 , width=self.sizecanv[0], height=self.sizecanv[1],background='white')
        self.canv.grid(row=2,column=0,columnspan=2)

        self.Drawlines()
        self.Drawbeams()

        self.canv.bind('<ButtonPress-1>', self.onCanvasClick)
        self.canv.bind('<ButtonPress-2>', self.onCanvasRightClick)
        self.var.trace('w', self.Changefile)
        self.canv.update()

        # beam list
        self.bidlabel = Label(self.frame2,text="Beam ID")
        self.bidlabel.grid(row=0,column=0)
        self.azlabel = Label(self.frame2,text="Azimuth")
        self.azlabel.grid(row=0,column=1)
        self.ellabel = Label(self.frame2,text="Elevation")
        self.ellabel.grid(row=0,column=2)

        self.scroll = Scrollbar(self.frame2)
        self.scroll.grid(row=1,column=3)

        self.beamtext = Text(self.frame2,yscrollcommand=self.scroll.set)
        self.beamtext.config(width=30,state=DISABLED)
        self.beamtext.grid(row = 1,column = 0,columnspan=3)

        self.scroll.config(command=self.beamtext.yview)

    def Changefile(self,*args):
        """ This function will change the files to a different radar system."""
        filename= self.choices[self.var.get()]
        self.beamtext.delete(1.0,END)
        self.readfile.set(filename)
        self.lines = np.loadtxt(filename)
        self.output=[]
        self.canv.delete(ALL)
        self.Drawlines()
        self.Drawbeams()

    def Drawbeams(self):
        "This function will draw all of the beams on the canvas."
        div =self.div
        #if self.beamhandle is not None:

        off_x = self.off_x
        off_y = self.off_y
        self.beamhandles = []
        self.ovalx = np.zeros(self.lines.shape[0])
        self.ovaly = np.zeros(self.lines.shape[0])
        for ibeam,beams in enumerate(self.lines):
            c_coords = rect(90-beams[2],beams[1]-90)
            #print c_coords
            points = [c_coords[0]*div/10+off_x -5,
                    c_coords[1]*div/10+off_y-5,
                    c_coords[0]*div/10+off_x +5,
                    c_coords[1]*div/10+off_y+5]
            self.ovalx[ibeam] = c_coords[0]*div/10+off_x
            self.ovaly[ibeam] = c_coords[1]*div/10+off_y
            self.beamhandles.append(self.canv.create_oval(points, fill='blue',tags='beams'))

    def Drawlines(self):
        """This function will draw all of the lines on the canvas for the """
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
            self.canv.create_text(points[0]+(div*i)/(div/15.0),points[1]+(div*i)/(div/15.0),text=str(lat[i-1]))
        for iang in angles:
            iangr = iang*np.pi/180
            cosan = np.cos(iangr)
            sinan = np.sin(iangr)

            points = [-div*Nlat*cosan+off_x, -div*Nlat*sinan+off_y, div*Nlat*cosan+off_x, div*Nlat*sinan+off_y]
            self.canv.create_line(points,dash=(5,5))
            self.canv.create_text(points[0],points[1], text=textangles[iang][0])
            self.canv.create_text(points[2],points[3], text=textangles[iang][1])

    def onCanvasClick(self,event):
        """This function will find the nearest beam """

        p_coords = polar(event.x-self.off_x ,event.y-self.off_y)
        x = self.canv.canvasx(event.x)
        y = self.canv.canvasy(event.y)
        dist = (self.ovalx-x)**2+(self.ovaly-y)**2
        linesit = np.argmin(dist)
        closest = self.lines[linesit]
        if closest[0] not in self.output:
            textheader = 'Closest beam is # %s, Az: %s, El: %s' %(int(closest[0]),closest[1],closest[2])
            self.canv.itemconfig(self.canv.find_withtag('header'),text=textheader)
            self.canv.itemconfig(self.beamhandles[linesit], fill='orange')

            self.output.append(closest[0])
            self.canv.update()

            self.beamtext.config(state=NORMAL)
            self.beamtext.insert(INSERT,"{:>9} {:>9} {:>9}\n".format(closest[0],closest[1],closest[2]))
            self.beamtext.config(state=DISABLED)
        else:
            print("Repeated beam")

    def buttonClick(self):
        """This will output the beam list, create an image of the beams and close the program. """
        allbeam = BeamSelector(self.lines)
        fig = allbeam.plotbeams(self.output,True,'beampic.png',"Chosenbeams")
        plt.close(fig)
        f = open('SelectedBeamCodes.txt', 'w')
        for beam in self.output:
            f.write("%s\n" % (int(beam)))
        sys.exit()

    def onCanvasRightClick(self,event):
        """This will undo the choice of the nearest highlighted beam."""
        x = self.canv.canvasx(event.x)
        y = self.canv.canvasy(event.y)
        dist = (self.ovalx-x)**2+(self.ovaly-y)**2
        linesit = np.argmin(dist)
        closest = self.lines[linesit]
        if (closest[0] in self.output) and (dist[linesit]<self.div/5.0):
            self.canv.itemconfig(self.beamhandles[linesit], fill='blue')
            self.output.remove(closest[0])
            self.canv.update()

def run_beam_gui():
    """Used to run the GUI as a function"""
    root = Tk()
    gui = Gui(root,False)
    root.mainloop()
if __name__ == "__main__":

    root = Tk()
    gui = Gui(root,False)
    root.mainloop()