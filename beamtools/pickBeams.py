#!/usr/bin/env python
"""
This GUI is based off of a GUI originally developed by Steven Chen at SRI.
The code was cleaned up so that the GUI is now encompassed in a class structure. Also
the ability to switch between PFISR and RISR-N has been added along with a finish button.
The code also outputs a picture of the selected beam pattern.

@author: John Swoboda
Updated by Greg Starr so it can be used as part of a larger GUI
"""

import Tkinter
import tkFileDialog
import os, inspect
import numpy as np
import matplotlib.pyplot as plt
import tables
from beamfuncs import BeamSelector
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
    def __init__(self,parent,subgui=True):
         # get the current path

        curpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        constpath = os.path.join(os.path.split(curpath)[0],'RadarDataSim','const')
        # set the root
        self.parent = parent
        self.subgui = subgui
        # set up frames for list
        self.frame1 = Tkinter.Frame(self.parent)
        self.frame1.grid(row=0,column=0)
        self.frame2 = Tkinter.Frame(self.parent)
        self.frame2.grid(row=0,column=1)

        self.output = []
        self.beamhandle = None
        if subgui:
            self.sizecanv = [500,500]
            self.beamcodeent= Tkinter.Entry(self.frame1)
            self.beamcodeent.grid(row=1,column=1)
            self.beamcodeentlabel = Tkinter.Label(self.frame1,text="Enter Beamcodes")
            self.beamcodeentlabel.grid(row=1,column=0,sticky='e')
            self.beambuttex = Tkinter.Button(self.frame1, text="Read", command=self.readbcobar)
            self.beambuttex.grid(row=1,column=2,sticky='w')
            self.beambutt = Tkinter.Button(self.frame1, text="Import", command=self.beambuttonClick)
            self.beambutt.grid(row=2,column=2,sticky='w')
            canvrow = 3
        else:
            self.sizecanv = [1000,1000]
            self.leb = Tkinter.Label(self.frame1, text="Beam Selector",font=("Helvetica", 16))
            self.leb.grid(row=0, sticky=Tkinter.W+Tkinter.E+Tkinter.N+Tkinter.S,columnspan=2)
            self.butt = Tkinter.Button(self.frame1, text="Finished", command=self.buttonClick)
            self.butt.grid(row=1,column=1,sticky='w')
            self.beamcodeent= Tkinter.Entry(self.frame1)
            self.beamcodeent.grid(row=2,column=1,sticky='w')
            self.beamcodeentlabel = Tkinter.Label(self.frame1,text="Enter Beamcodes")
            self.beamcodeentlabel.grid(row=2,column = 0,sticky='e')
            self.beambuttex = Tkinter.Button(self.frame1, text="Read", command=self.readbcobar)
            self.beambuttex.grid(row=2,column=2,sticky='w')
            self.beambutt = Tkinter.Button(self.frame1, text="Import", command=self.beambuttonClick)
            self.beambutt.grid(row=3,column=2,sticky='w')

            canvrow = 4
        self.off_x = self.sizecanv[0]/2
        self.off_y = self.sizecanv[1]/2
        self.div = 75.0*self.sizecanv[0]/1000.0
        self.lat = [80,70,60,50,40,30]
        self.angles = np.arange(0,180,30)

        self.var = Tkinter.StringVar()
        self.var.set("PFISR")
        self.choices = {"PFISR":os.path.join(constpath,'PFISR_PARAMS.h5'),
                        "RISR-N":os.path.join(constpath,'RISR_PARAMS.h5'),
                        "Sondrestrom":os.path.join(constpath,'Sondrestrom_PARAMS.h5'),
                        "Millstone":os.path.join(constpath,'Millstone_PARAMS.h5')}#, "RISR-S":'file3'}
        self.option = Tkinter.OptionMenu(self.frame1, self.var, *self.choices)
        self.option.grid(row=1,column=0,sticky='w')
        hfile=tables.open_file(self.choices[self.var.get()])
        self.lines = hfile.root.Params.Kmat.read()
        hfile.close()
        self.readfile = Tkinter.StringVar()

        # set up the canvas
        self.canv = Tkinter.Canvas(self.frame1 , width=self.sizecanv[0], height=self.sizecanv[1],background='white')
        self.canv.grid(row=canvrow,column=0,columnspan=2)

        self.Drawlines()
        self.Drawbeams()

        self.canv.bind('<ButtonPress-1>', self.onCanvasClick)
        self.canv.bind('<ButtonPress-2>', self.onCanvasRightClick)
        self.var.trace('w', self.Changefile)
        self.canv.update()

        # beam list
        self.bidlabel = Tkinter.Label(self.frame2,text="Beam ID")
        self.bidlabel.grid(row=0,column=0)
        self.azlabel = Tkinter.Label(self.frame2,text="Azimuth")
        self.azlabel.grid(row=0,column=1)
        self.ellabel = Tkinter.Label(self.frame2,text="Elevation")
        self.ellabel.grid(row=0,column=2)

        self.scroll = Tkinter.Scrollbar(self.frame2)
        self.scroll.grid(row=1,column=3)

        self.beamtext = Tkinter.Text(self.frame2,yscrollcommand=self.scroll.set)
        self.beamtext.config(width=50,state=Tkinter.DISABLED)
        self.beamtext.grid(row = 1,column = 0,columnspan=3)
        self.beamlines = []
        self.scroll.config(command=self.beamtext.yview)
        
        # bounding box
        self.boxbutton= Tkinter.Button(self.frame2, text="Angle Box", command=self.boxbuttonClick)
        self.boxbutton.grid(row=2,column=0,sticky='w')
        
        self.azminmaxlabel = Tkinter.Label(self.frame2,text="Az min and max")
        self.azminmaxlabel.grid(row=3,column=0,sticky='e')
        self.azmin= Tkinter.Entry(self.frame2)
        self.azmin.grid(row=3,column=1,sticky='w')
        self.azmax= Tkinter.Entry(self.frame2)
        self.azmax.grid(row=3,column=2,sticky='w')
        
        self.elminmaxlabel = Tkinter.Label(self.frame2,text="El min and max")
        self.elminmaxlabel.grid(row=4,column=0,sticky='e')
        self.elmin= Tkinter.Entry(self.frame2)
        self.elmin.grid(row=4,column=1,sticky='w')
        self.elmax= Tkinter.Entry(self.frame2)
        self.elmax.grid(row=4,column=2,sticky='w')
        
        # Az choice
        self.azbutton=Tkinter.Button(self.frame2, text="Az Choice", command=self.azbuttonClick)
        self.azbutton.grid(row=5,column=0,sticky='w')
        self.azchoice= Tkinter.Entry(self.frame2)
        self.azchoice.grid(row=5,column=1,sticky='w')
        
        # Az choice
        self.elbutton=Tkinter.Button(self.frame2, text="El Choice", command=self.elbuttonClick)
        self.elbutton.grid(row=6,column=0,sticky='w')
        self.elchoice= Tkinter.Entry(self.frame2)
        self.elchoice.grid(row=6,column=1,sticky='w')
        
        self.azsortbutton=Tkinter.Button(self.frame2, text="Az sort", command=self.azsortbuttonClick)
        self.azsortbutton.grid(row=7,column=0,sticky='w')
        self.elsortbutton=Tkinter.Button(self.frame2, text="El Sort", command=self.elsortbuttonClick)
        self.elsortbutton.grid(row=7,column=1,sticky='w')

    
        
    def Changefile(self,*args):
        """ This function will change the files to a different radar system."""
        filename= self.choices[self.var.get()]
        self.beamtext.config(state=Tkinter.NORMAL)
        self.beamtext.delete(1.0,'end')
        self.beamtext.config(state=Tkinter.DISABLED)
        self.readfile.set(filename)
        hfile=tables.open_file(filename)
        self.lines = hfile.root.Params.Kmat.read()
        hfile.close()
        self.output=[]
        self.canv.delete(Tkinter.ALL)
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

    def addbeamlist(self,beamlist):
        """ """
        div =self.div
        off_x = self.off_x
        off_y = self.off_y
        for ibeam in beamlist:
            c_coords = rect(90-ibeam[1],ibeam[0]-90)
            x = c_coords[0]*div/10+off_x
            y = c_coords[1]*div/10+off_y

            dist = (self.ovalx-x)**2+(self.ovaly-y)**2
            linesit = np.argmin(dist)
            closest = self.lines[linesit]
            if closest[0] not in self.output:
                self.__addbeam__(closest,linesit)
    def addbeamlistbco(self,bcolist):
        """ Adds a set of beams based off of the beam numbers"""
        allbco = self.lines[:,0]
        allbco = np.array([int(i) for i in allbco])

        for ibco in bcolist:
            ibco = int(ibco)
            linesit =np.flatnonzero(allbco==ibco)
            if len(linesit)==0:
                continue
            linesit = linesit[0]
            closest= self.lines[linesit]
            if closest[0] not in self.output:
                self.__addbeam__(closest,linesit)
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
            self.__addbeam__(closest,linesit)
        else:
            print("Repeated beam")
    def onCanvasRightClick(self,event):
        """This will undo the choice of the nearest highlighted beam."""
        x = self.canv.canvasx(event.x)
        y = self.canv.canvasy(event.y)
        dist = (self.ovalx-x)**2+(self.ovaly-y)**2
        linesit = np.argmin(dist)
        closest = self.lines[linesit]
        if (closest[0] in self.output) and (dist[linesit]<self.div/5.0):
            self.__removebeam__(closest,linesit)
    
    def boxbuttonClick(self):
        """This the call back for the bounding box button where all of the beams
        at a certian elevation are selected."""
        inputvec = []
        inputvec.append(self.azmin.get().strip())
        inputvec.append(self.azmax.get().strip())
        inputvec.append(self.elmin.get().strip())
        inputvec.append(self.elmax.get().strip())
        maxmin = [0.,359.99,0.,90.]
        
        inputnums = []
        for i,iin in enumerate(inputvec):
            try:
                inputnums.append(float(iin))
            except:
                inputnums.append(maxmin(i))
               
        
        alldata = self.lines
        azkeep = np.logical_and(alldata[:,1]>inputnums[0],alldata[:,1]<inputnums[1])
        elkeep = np.logical_and(alldata[:,2]>inputnums[2],alldata[:,2]<inputnums[3])
        allkeep = np.logical_and(azkeep,elkeep)        
        bcolist = alldata[allkeep,0]
        if (len(bcolist)!=0) and (len(bcolist)!=len(alldata)):        
            self.addbeamlistbco(bcolist)
        
    def azbuttonClick(self):
        """This the call back for the azimuth button where all of the beams
        at a certian elevation are selected."""
        azval = self.azchoice.get().strip()
        try:
            aznum = float(azval)
            azkeep =np.in1d(self.lines[:,1],aznum)
            bcoout = self.lines[azkeep,0]
            self.addbeamlistbco(bcoout)
        except:
            print('Bad value for azimuth angle given')
    def elbuttonClick(self):
        """This the call back for the elevation button where all of the beams
        at a certian elevation are selected."""
        elval = self.elchoice.get().strip()
        try:
            elnum = float(elval)
            elkeep =np.in1d(self.lines[:,2],elnum)
            bcoout = self.lines[elkeep,0]
            self.addbeamlistbco(bcoout)
        except:
            print('Bad value for elevation angle given')
            
    def __removebeam__(self,closest,linesit):
        """This removes a beam from the data"""
        self.canv.itemconfig(self.beamhandles[linesit], fill='blue')
        self.output.remove(closest[0])
        beamstr = "{:>9} {:>9} {:>9}\n".format(closest[0],closest[1],closest[2])
        self.beamlines.remove(beamstr)
        self.beamtext.config(state=Tkinter.NORMAL)
        self.beamtext.delete(1.0,'end')
        for ibeam in self.beamlines:
            self.beamtext.insert(Tkinter.INSERT,ibeam)

        self.beamtext.config(state=Tkinter.DISABLED)
        self.canv.update()

    def azsortbuttonClick(self):
        outlist = self.output
        bmlist = self.lines
        bcolist = bmlist[:,0]
        azlist = bmlist[:,1]
        azvals = [azlist[bcolist==i][0] for i in outlist ]
        order = np.argsort(azvals)
        self.updatelists(order)
    def elsortbuttonClick(self):
        outlist = self.output
        bmlist = self.lines
        bcolist = bmlist[:,0]
        ellist = bmlist[:,2]
        elvals = [ellist[bcolist==i][0] for i in outlist ]
        order = np.argsort(elvals)
        self.updatelists(order)
    def __addbeam__(self,closest,linesit):
        """This will add a beam"""
        textheader = 'Closest beam is # %s, Az: %s, El: %s' %(int(closest[0]),closest[1],closest[2])
        self.canv.itemconfig(self.canv.find_withtag('header'),text=textheader)
        self.canv.itemconfig(self.beamhandles[linesit], fill='orange')

        self.output.append(closest[0])
        self.canv.update()

        beamstr = "{:>9} {:>9} {:>9}\n".format(closest[0],closest[1],closest[2])
        self.beamtext.config(state=Tkinter.NORMAL)
        self.beamtext.insert(Tkinter.INSERT,beamstr)
        self.beamtext.config(state=Tkinter.DISABLED)
        self.beamlines.append(beamstr)
        bcolist = self.beamcodeent.get().split()
        bcolist = [int(i.strip(',')) for i in bcolist]
        cbco = int(closest[0])
        if cbco not in bcolist:
            bcolist.append(cbco)
            bcoliststr = [str(ib) for ib in bcolist]
            bcostr = ' '.join(bcoliststr)
            self.beamcodeent.delete(0,'end')
            self.beamcodeent.insert(0,bcostr)
    
    def removebeamlistbco(self,bcolist):
        """ Removes a set of beams based off of the beam numbers"""
        allbco = self.lines[:,0]
        allbco = np.array([int(i) for i in allbco])

        for ibco in bcolist:
            ibco = int(ibco)
            linesit =np.flatnonzero(allbco==ibco)
            if len(linesit)==0:
                continue
            linesit = linesit[0]
            closest= self.lines[linesit]
            if closest[0] in self.output:
                self.__removebeam__(closest,linesit)
    def updatelists(self,order):
        neworder =[self.output[i] for i in order]
        self.removebeamlistbco(neworder)
        self.addbeamlistbco(neworder)
    def buttonClick(self,fn=None):
        """This will output the beam list, create an image of the beams and close the program. """
        if fn is None:
            fn = tkFileDialog.asksaveasfilename(title="save Beam Codes and Image",filetypes=[('TXT','.txt')])
        fnbase = ''.join(os.path.splitext(fn)[:-1])
        allbeam = BeamSelector(self.lines)
        fig = allbeam.plotbeams(self.output,True,fnbase+'.png',"Chosenbeams")
        plt.close(fig)
        f = open(fnbase+'.txt', 'w')
        for beam in self.output:
            f.write("%s\n" % (int(beam)))
        if not self.subgui:
            #sys.exit()
            self.parent.destroy()

    def beambuttonClick(self):
        fn = tkFileDialog.askopenfilename(title="Load Beam Codes",filetypes=[('TXT','.txt')])
        bcolist = np.loadtxt(fn)
        self.addbeamlistbco(bcolist)

    def readbcobar(self):
        bcolist = self.beamcodeent.get().split()
        bcolist = [int(i.strip(',')) for i in bcolist]
        self.addbeamlistbco(bcolist)

def run_beam_gui():
    """Used to run the GUI as a function"""
    root = Tkinter.Tk()
    gui = Gui(root,False)
    root.mainloop()
if __name__ == "__main__":

    root = Tkinter.Tk()
    gui = Gui(root,False)
    root.mainloop()