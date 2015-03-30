from Tkinter import *
import numpy

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

def onCanvasClick(event):
    if event.widget.find_closest(event.x, event.y) == canv.find_withtag('finish'):
        print 'Finished'
        print canv.output
        f = open('SelectedBeamCodes.txt', 'w')
        for beam in canv.output:
            f.write("%s\n" % (beam))
        sys.exit()
    else:
        p_coords = polar(event.x-500,event.y-500)
        print  'Clicked Az: ', p_coords[1],'El: ', numpy.abs(90 - p_coords[0]/6)
        fname = 'bcotable.txt'
        lines = numpy.loadtxt('bcotable.txt')
        closest = lines[numpy.int32(event.widget.find_closest(event.x, event.y))-1]
        canv.itemconfig(canv.find_withtag('header'),text='Closest beam is # %s, Az: %s, El: %s' %(closest[0][0],closest[0][1],closest[0][2]))
        canv.itemconfig(event.widget.find_closest(event.x, event.y), fill='orange')
        print 'Closest beam is # %s, Az: %s, El: %s' %(closest[0][0],closest[0][1],closest[0][2])
        canv.output.append(closest[0][0])



def onCanvasRightClick(event):
    if numpy.int32(event.widget.find_closest(event.x, event.y)) < 481:
        canv.itemconfig(event.widget.find_closest(event.x, event.y), fill='blue')
        closest = lines[numpy.int32(event.widget.find_closest(event.x, event.y))-1]
        if closest[0][0] in canv.output:
            canv.output.remove(closest[0][0])


# divide by 4


root = Tk()
root.title("Beam Selector")
canv = Canvas(root, width=1000, height=1000)
canv.output = []
offset_x = 500
offset_y = 500

# Create circles

div = 60
lat = [80,70,60,50,40,30]

#create dots
fname = 'bcotable.txt'
lines = numpy.loadtxt('bcotable.txt')

#now draw the dots -- needed to rotate by 90 degrees counter-clockwise
for beams in lines:
    c_coords = rect(90-beams[2],beams[1]-90)
    print c_coords
    points = [c_coords[0]*div/10+offset_x -5,
            c_coords[1]*div/10+offset_y-5,
            c_coords[0]*div/10+offset_x +5,
            c_coords[1]*div/10+offset_y+5]
    canv.create_oval(points, fill='blue')

canv.create_text(500,100, text='', tags = 'header')

for i in range(1,7):
    points = [-div*i+offset_x, -div*i+offset_y, div*i+offset_x, div*i+offset_y]
    canv.create_oval(points,dash=(5,5))
    canv.create_text(points[0]+(div*i)/(div/15),points[1]+(div*i)/(div/15),text=str(lat[i-1]))


#draw lines
points = [offset_x, -div*len(lat)+offset_y, offset_x, div*len(lat)+offset_y]
canv.create_line(points,dash=(5,5))
canv.create_text(points[0],points[1], text='0')
canv.create_text(points[2],points[3], text='180')
points = [-div*len(lat)+offset_x, offset_y, div*len(lat)+offset_x, offset_y]
canv.create_line(points,dash=(5,5))
canv.create_text(points[0],points[1], text='270')
canv.create_text(points[2],points[3], text='90')

angle = 30
points = [-div*len(lat)*numpy.cos(angle*(numpy.pi/180))+offset_x, -div*len(lat)*numpy.sin(angle*(numpy.pi/180))+offset_y, div*len(lat)*numpy.cos(angle*(numpy.pi/180))+offset_x, div*len(lat)*numpy.sin(angle*(numpy.pi/180))+offset_y]
canv.create_line(points,dash=(5,5))
canv.create_text(points[0],points[1], text='300')
canv.create_text(points[2],points[3], text='120')

angle = 60
points = [-div*len(lat)*numpy.cos(angle*(numpy.pi/180))+offset_x, -div*len(lat)*numpy.sin(angle*(numpy.pi/180))+offset_y, div*len(lat)*numpy.cos(angle*(numpy.pi/180))+offset_x, div*len(lat)*numpy.sin(angle*(numpy.pi/180))+offset_y]
canv.create_line(points,dash=(5,5))
canv.create_text(points[0],points[1], text='330')
canv.create_text(points[2],points[3], text='150')

angle = 120
points = [-div*len(lat)*numpy.cos(angle*(numpy.pi/180))+offset_x, -div*len(lat)*numpy.sin(angle*(numpy.pi/180))+offset_y, div*len(lat)*numpy.cos(angle*(numpy.pi/180))+offset_x, div*len(lat)*numpy.sin(angle*(numpy.pi/180))+offset_y]
canv.create_line(points,dash=(5,5))
canv.create_text(points[0],points[1], text='30')
canv.create_text(points[2],points[3], text='210')

angle = 150
points = [-div*len(lat)*numpy.cos(angle*(numpy.pi/180))+offset_x, -div*len(lat)*numpy.sin(angle*(numpy.pi/180))+offset_y, div*len(lat)*numpy.cos(angle*(numpy.pi/180))+offset_x, div*len(lat)*numpy.sin(angle*(numpy.pi/180))+offset_y]
canv.create_line(points,dash=(5,5))

canv.create_text(points[0],points[1], text='60')
canv.create_text(points[2],points[3], text='240')


canv.create_text(900,900,text = 'Done', tags='finish')




canv.bind('<ButtonPress-1>', onCanvasClick)
canv.bind('<ButtonPress-2>', onCanvasRightClick)
canv.pack()
root.mainloop()
