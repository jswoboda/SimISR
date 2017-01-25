#!/usr/bin/env python
"""
This GUI can be used to create set up files for the SimISR. The user can set up
the parameters and set up the beam pattern. The user can also bring in an older setup
file, change the settings and then save out a new version.

@author: Greg Starr
"""
#from Tkinter import *
import Tkinter as Tk
import tkFileDialog
import pickBeams as pb
import pdb
import scipy as sp
from SimISR.utilFunctions import makeconfigfile,readconfigfile

class App():

    def __init__(self,root):
        self.root = root
        self.root.title("SimISR")
        # title
        self.titleframe = Tk.Frame(self.root)
        self.titleframe.grid(row=0,columnspan=3)
        self.menubar = Tk.Menu(self.titleframe)
        # filemenu stuff
        self.filemenu = Tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Load", command=self.loadfile)
        self.filemenu.add_command(label="Save", command=self.savefile)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.root.config(menu=self.menubar)
        # frame label
        self.frame = Tk.LabelFrame(self.root, text="Sim Params", padx=5, pady=5)
        self.frame.grid(row=1,column=0, sticky="e")
        #Gui label
        self.leb = Tk.Label(self.titleframe, text="Radar Data Sim GUI",font=("Helvetica", 16))
        self.leb.grid()

        rown = 4
        #IPP stuff
        self.ipp = Tk.Entry(self.frame)
        self.ipp.grid(row=rown,column=1)
        self.ipplabel = Tk.Label(self.frame,text="IPP (sec)")
        self.ipplabel.grid(row=rown,column=0)
        rown+=1
        # Range limits
        self.rangelimlow = Tk.Entry(self.frame)
        self.rangelimhigh = Tk.Entry(self.frame)
        self.rangelimlow.grid(row=rown,column=1)
        self.rangelimhigh.grid(row=rown,column=2)
        self.rangelabel = Tk.Label(self.frame,text="Range Gate Limits (km)")
        self.rangelabel.grid(row=rown)
        rown+=1
        # pulse length
        self.pulselength = Tk.Entry(self.frame)
        self.pulselength.grid(row=rown,column=1)
        self.pulselengthlabel = Tk.Label(self.frame,text="Pulse Length (us)")
        self.pulselengthlabel.grid(row=rown)
        rown+=1
        # Sampling rate
        self.t_s = Tk.Entry(self.frame)
        self.t_s.grid(row=rown,column=1)
        self.t_slabel = Tk.Label(self.frame,text="Sampling Time (us)")
        self.t_slabel.grid(row=rown)
        rown+=1
        # Pulse type update
        self.pulsetype = Tk.StringVar()
        self.pulsetype.set("Long")
        self.pulsetypelabel = Tk.Label(self.frame,text="Pulse Type")
        self.pulsetypelabel.grid(row=rown)
        self.pulsetypemenu = Tk.OptionMenu(self.frame, self.pulsetype,"Long","Barker",command=self.set_defaults)
        self.pulsetypemenu.grid(row=rown,column=1,sticky='w')
        rown+=1
        # Integration Time
        self.tint = Tk.Entry(self.frame)
        self.tint.grid(row=rown,column=1)
        self.tintlabel = Tk.Label(self.frame,text="Integration time (s)")
        self.tintlabel.grid(row=rown)
        rown+=1
        # Fitter time interval
        self.fitinter = Tk.Entry(self.frame)
        self.fitinter.grid(row=rown,column=1)
        self.fitinterlabel = Tk.Label(self.frame,text="Time interval between fits (s)")
        self.fitinterlabel.grid(row=rown)
        rown+=1
        # Fitter time Limit
        self.timelim = Tk.Entry(self.frame)
        self.timelim.grid(row=rown,column=1)
        self.timelimlabel = Tk.Label(self.frame,text="Simulation Time limit (s)")
        self.timelimlabel.grid(row=rown)
        rown+=1
        # Number of noise samples per pulse
        self.nns = Tk.Entry(self.frame)
        self.nns.grid(row=rown,column=1)
        self.nnslabel = Tk.Label(self.frame,text="noise samples per pulse")
        self.nnslabel.grid(row=rown)
        rown+=1
        # Number of noise pulses
        # XXX May get rid of this
        self.nnp = Tk.Entry(self.frame)
        self.nnp.grid(row=rown,column=1)
        self.nnplabel = Tk.Label(self.frame,text="number of noise pulses")
        self.nnplabel.grid(row=rown)
        rown+=1
        # Data type
        self.dtype = Tk.StringVar()
        self.dtype.set("complex128")
        self.dtypelabel = Tk.Label(self.frame,text="Raw Data Type")
        self.dtypelabel.grid(row=rown)
        self.dtypemenu = Tk.OptionMenu(self.frame, self.dtype,"complex64","complex128")
        self.dtypemenu.grid(row=rown,column=1,sticky='w')
        rown+=1
        # Upsampling factor for the ambiguity funcition
        # XXX May get rid of this.
        self.ambupsamp = Tk.Entry(self.frame)
        self.ambupsamp.grid(row=rown,column=1)
        self.ambupsamplabel = Tk.Label(self.frame,text="Up sampling factor for ambiguity function")
        self.ambupsamplabel.grid(row=rown)
        rown+=1
        # Species
        self.species = Tk.Entry(self.frame)
        self.species.grid(row=rown,column=1)
        self.specieslabel = Tk.Label(self.frame,text="Species N2+, N+, O+, NO+, H+, O2+, e-")
        self.specieslabel.grid(row=rown)
        rown+=1
        # Number of samples per spectrum
        self.numpoints = Tk.Entry(self.frame)
        self.numpoints.grid(row=rown,column=1)
        self.numpointslabel = Tk.Label(self.frame,text="Number of Samples for Sectrum")
        self.numpointslabel.grid(row=rown)
        rown+=1
        # Start file for set up
        self.startfile = Tk.Entry(self.frame)
        self.startfile.grid(row=rown,column=1)
        self.startfilelabel = Tk.Label(self.frame,text="Start File")
        self.startfilelabel.grid(row=rown)
        rown+=1
        # Fitting Type
        self.fittype = Tk.StringVar()
        self.fittype.set("Spectrum")
        self.fittypelabel = Tk.Label(self.frame,text="Fit type")
        self.fittypelabel.grid(row=rown)
        self.fittypemenu = Tk.OptionMenu(self.frame, self.fittype,"Spectrum","ACF")
        self.fittypemenu.grid(row=rown,column=1,sticky='w')
        rown+=1
                # outangles output
        self.outangles = Tk.Entry(self.frame)
        self.outangles.grid(row=rown,column=1)
        self.outangleslabel = Tk.Label(self.frame,text="Beam int together, seperated by commas")
        self.outangleslabel.grid(row=rown)

        # Beam selector GUI
        self.frame2 = Tk.LabelFrame(self.root,text="Beam Selector",padx=5,pady=5)
        self.frame2.grid(row=1,column=1, sticky="e")

        self.pickbeams = pb.Gui(self.frame2)

#        self.timelim=DoubleVar()
        self.set_defaults()
        self.paramdic =   {'IPP':self.ipp,
                           'TimeLim':self.timelim,
                           'RangeLims':[self.rangelimlow,self.rangelimhigh],
                           'Pulselength':self.pulselength,
                           't_s': self.t_s,
                           'Pulsetype':self.pulsetype,
                           'Tint':self.tint,
                           'Fitinter':self.fitinter,
                           'NNs': self.nns,
                           'NNp':self.nnp,
                           'dtype':self.dtype,
                           'ambupsamp':self.ambupsamp,
                           'species':self.species,
                           'numpoints':self.numpoints,
                           'startfile':self.startfile,
                           'FitType':self.fittype }

    def set_defaults(self,*args):
        """Set the default files for the data."""
        self.ipp.delete(0, 'end')
        self.ipp.insert(0,'8.7e-3')

        self.tint.delete(0,'end')
        self.tint.insert(0,'180')
        self.fitinter.delete(0,'end')
        self.fitinter.insert(0,'180')

        self.species.delete(0, 'end')
        self.species.insert(0,'O+ e-')
        self.numpoints.delete(0, 'end')
        self.numpoints.insert(0,'128')
        self.ambupsamp.delete(0, 'end')
        self.ambupsamp.insert(0,'1')
        self.timelim.delete(0, 'end')
        self.timelim.insert(0,'540')
        # noise
        self.nns.delete(0,'end')
        self.nns.insert(0,'28')
        self.nnp.delete(0,'end')
        self.nnp.insert(0,'100')
        # For different pulse types
        self.rangelimlow.delete(0, 'end')
        self.rangelimhigh.delete(0, 'end')
        self.pulselength.delete(0, 'end')
        self.t_s.delete(0, 'end')


        if self.pulsetype.get().lower()=='long':
            self.rangelimlow.insert(0,'150')
            self.rangelimhigh.insert(0,'500')
            self.pulselength.insert(0,'280')
            self.t_s.insert(0,'20')
        elif self.pulsetype.get().lower()=='barker':
            self.rangelimlow.insert(0,'50')
            self.rangelimhigh.insert(0,'150')
            self.t_s.insert(0,'10')
            self.pulselength.insert(0,'130')

    def savefile(self):
        """Saves the parameters out"""
        fn = tkFileDialog.asksaveasfilename(title="Save File",filetypes=[('INI','.ini'),('PICKLE','.pickle')])
        blist = self.pickbeams.output
        self.pickbeams.buttonClick(fn)
        radarname = self.pickbeams.var.get()
        posspec =  ['N2+', 'N+', 'O+', 'NO+', 'H+', 'O2+','e-' ]
        specieslist = self.species.get().lower().split()
        newlist =[x for x in posspec if x.lower() in specieslist]

        if 'e-' not in newlist:newlist.append('e-')
        
        
        simparams ={'IPP':float(self.ipp.get()),
                    'TimeLim':float(self.timelim.get()),
                    'RangeLims':[int(float(self.rangelimlow.get())),int(float(self.rangelimhigh.get()))],
                    'Pulselength':1e-6*float(self.pulselength.get()),
                    't_s': 1e-6*float(self.t_s.get()),
                    'Pulsetype':self.pulsetype.get(),
                    'Tint':float(self.tint.get()),
                    'Fitinter':float(self.fitinter.get()),
                    'NNs': int(float(self.nns.get())),
                    'NNp':int(float(self.nnp.get())),
                    'dtype':{'complex128':sp.complex128,'complex64':sp.complex64}[self.dtype.get()],
                    'ambupsamp':int(float(self.ambupsamp.get())),
                    'species':newlist,
                    'numpoints':int(float(self.numpoints.get())),
                    'startfile':self.startfile.get(),
                    'FitType': self.fittype.get()}
        
        if len(self.outangles.get())>0:
            outlist1 = self.outangles.get().split(',')
            simparams['outangles']=[[ float(j) for j in  i.lstrip().rstrip().split(' ')] for i in outlist1]
                    
        makeconfigfile(fn,blist,radarname,simparams)


    def loadfile(self):
        """Imports parameters from old files"""
        fn = tkFileDialog.askopenfilename(title="Load File",filetypes=[('INI','.ini'),('PICKLE','.pickle')])
        try:
            sensdict,simparams = readconfigfile(fn)
            rdrnames = {'PFISR':'PFISR','pfisr':'PFISR','risr':'RISR-N','RISR-N':'RISR-N','RISR':'RISR-N'}
            currdr = rdrnames[sensdict['Name']]
            fitnfound = True
            for i in simparams:
                try:
                    if i=='RangeLims':
                        self.paramdic[i][0].delete(0,Tk.END)
                        self.paramdic[i][1].delete(0,Tk.END)
                        self.paramdic[i][0].insert(0,str(simparams[i][0]))
                        self.paramdic[i][1].insert(0,str(simparams[i][1]))
                    elif i=='species':
                        self.paramdic[i].delete(0,Tk.END)
                        string=''
                        if isinstance(simparams[i],list):
                            for a in simparams[i]:
                                string+=a
                                string+=" "
                        else:
                            string = simparams[i]
                        self.paramdic[i].insert(0,string)
                    elif i=='Pulselength' or i=='t_s':
                        self.paramdic[i].delete(0,Tk.END)
                        num = float(simparams[i])*10**6
                        self.paramdic[i].insert(0,str(num))
                    elif i== 'FitType':
                        self.fittype = simparams[i]
                        fitnfound=False
                    else:
                        self.paramdic[i].delete(0,Tk.END)
                        self.paramdic[i].insert(0,str(simparams[i]))
                except:
                    if simparams[i]==sp.complex128:
                        self.paramdic[i].set('complex128')
                    elif simparams[i]==sp.complex64:
                        self.paramdic[i].set('complex64')
                    elif i in self.paramdic:
                        self.paramdic[i].set(simparams[i])
            if fitnfound:
                self.fittype = 'Spectrum'
            self.pickbeams.var.set(currdr)
            self.pickbeams.Changefile()
            self.pickbeams.addbeamlist(simparams['angles'])
        except:
            print "Failed to import file."




def runsetupgui():

    root = Tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":

    root = Tk.Tk()
    app = App(root)
    root.mainloop()
