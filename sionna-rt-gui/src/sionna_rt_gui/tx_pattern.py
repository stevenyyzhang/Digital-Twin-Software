"""Camera-aligned discrete selection tiles; tiles are not physical beam widths."""
import numpy as np
import polyscope as ps
import polyscope.imgui as ui
from .beamforming import rotation
from .panel_codebook import PANEL_ROTATIONS, PANEL_NAMES, SIDE, PER_PANEL

def beam_tiles():
    edges=np.linspace(-1.,1.,SIDE+1);tiles=[]
    for rot in PANEL_ROTATIONS:
        for row in range(SIDE):
            for col in range(SIDE):
                q=np.array([[1,edges[col],edges[row]],[1,edges[col+1],edges[row]],
                            [1,edges[col+1],edges[row+1]],[1,edges[col],edges[row+1]]])
                tiles.append((q/np.linalg.norm(q,axis=1,keepdims=True))@rot.T)
    return np.asarray(tiles)

def front_polygon(points):
    result=[]
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        if a[2]>=0:result.append(a)
        if (a[2]>=0)!=(b[2]>=0):result.append(a+(b-a)*a[2]/(a[2]-b[2]))
    return np.asarray(result)

def inside_polygon(point,polygon):
    edge=np.roll(polygon,-1,axis=0)-polygon;delta=point-polygon
    cross=edge[:,0]*delta[:,1]-edge[:,1]*delta[:,0]
    return bool(np.all(cross>=-1e-6) or np.all(cross<=1e-6))

def beam_at_direction(direction):
    # Intersect the sphere direction with its owning cube face. This avoids
    # ambiguous polygon overlap at the silhouette of the projected sphere.
    local=np.asarray(direction)@PANEL_ROTATIONS
    panel=int(np.argmax(local[:,0]))
    uv=local[panel,1:]/local[panel,0]
    col,row=np.clip(np.floor((uv+1)*SIDE/2).astype(int),0,SIDE-1)
    return int(panel*PER_PANEL+row*SIDE+col)

def axis_label_positions(anchors,sizes,start,side):
    """Keep complete labels inside the selector, separating projected axes."""
    lo=np.asarray(start)+6;hi=np.asarray(start)+side-6
    placed=[]
    for anchor,size in zip(anchors,sizes):
        size=np.asarray(size);base=np.asarray(anchor)-size/2
        step=size[1]+8
        candidates=[np.clip(base+offset,lo,hi-size) for offset in
                    ((0,0),(0,step),(0,-step),(size[0]+8,0),(-size[0]-8,0),
                     (0,2*step),(0,-2*step))]
        # Near a corner, clamping several offsets can produce the same point.
        # A bounded fallback grid also handles nearly coincident projected axes.
        grid=[np.array([x,y]) for x in np.linspace(lo[0],hi[0]-size[0],3)
                             for y in np.linspace(lo[1],hi[1]-size[1],3)]
        candidates+=sorted(grid,key=lambda p:np.linalg.norm(p-base))
        for point in candidates:
            if all(np.any(point+size+4<=p) or np.any(p+s+4<=point) for p,s in placed):break
        placed.append((point,size))
    return [p for p,_ in placed]

class TxPattern:
    def __init__(self):self.tiles=beam_tiles();self.drag=0
    def draw(self,app,compact=False):
        tx=app.scene.get('tx-0')
        if tx is None:return
        b=app.beam_demo
        if not compact:ui.Begin('Beam selector')
        ui.TextColored((.5,.9,.8,1),b.label)
        side=ui.GetContentRegionAvail()[0]
        start=np.asarray(ui.GetCursorScreenPos());center=start+side/2;radius=side*.43
        ui.InvisibleButton('Discrete beam sphere',(side,side))
        hovered=ui.IsItemHovered();io=ui.GetIO()
        if hovered and ui.IsMouseClicked(0):self.drag=0
        if ui.IsItemActive():self.drag+=sum(abs(x) for x in io.MouseDelta)
        camera=ps.get_view_camera_parameters()
        self.camera_basis=np.stack([camera.get_right_dir(),-np.asarray(camera.get_up_dir()),
                                    -np.asarray(camera.get_look_dir())],axis=1)
        projected=self.tiles@rotation(tx).T@self.camera_basis
        palette=np.array([[.90,.55,.41],[.45,.77,.65],[.61,.58,.87],
                          [.40,.70,.88],[.91,.76,.42],[.86,.52,.72]])
        draw=ui.GetWindowDrawList();hit=None;self.visible_tiles={}
        mouse=(np.asarray(io.MousePos)-center)/radius
        if hovered and mouse@mouse<=1:
            view=np.r_[mouse,np.sqrt(max(0.,1-mouse@mouse))]
            hit=beam_at_direction(view@self.camera_basis.T@rotation(tx))
        for i in np.argsort(projected[:,:,2].mean(axis=1)):
            poly=front_polygon(projected[i])
            if len(poly)<3:continue
            xy=poly[:,:2]*radius+center
            a,z=xy[1]-xy[0],xy[2]-xy[0]
            if abs(a[0]*z[1]-a[1]*z[0])<1e-5:continue
            self.visible_tiles[int(i)]=xy
            active=i==b.selected;over=i==hit
            rgb=np.array([1.,.92,.42]) if active else palette[i//PER_PANEL]*(1.12 if over else .88+(.09 if (i%SIDE+i//SIDE)%2 else 0))
            r,g,blue=np.clip(rgb*255,0,255).astype(int)
            color=int(r)|(int(g)<<8)|(int(blue)<<16)|(255<<24)
            for j in range(1,len(xy)-1):draw.AddTriangleFilled(tuple(xy[0]),tuple(xy[j]),tuple(xy[j+1]),color)
            for a,z in zip(xy,np.roll(xy,-1,axis=0)):
                draw.AddLine(tuple(a),tuple(z),0xFFFFFFFF if active else 0xFF332C25,2. if active else 1.)
            if active:draw.AddText(tuple(xy.mean(axis=0)-[8,6]),0xFF151515,str(i))
        if hit is not None:
            ui.SetTooltip(f'Beam {hit} | {PANEL_NAMES[hit//PER_PANEL]} | click to select')
            if ui.IsMouseReleased(0) and self.drag<=3:b.select(hit)
        labels=('+X','+Y','+Z')
        anchors=center+(np.eye(3)@self.camera_basis)[:,:2]*radius*1.08
        sizes=[np.asarray(ui.CalcTextSize(label)) for label in labels]
        positions=axis_label_positions(anchors,sizes,start,side)
        for label,anchor,point,size in zip(labels,anchors,positions,sizes):
            draw.AddLine(tuple(anchor),tuple(point+size/2),0xFFFFFFFF,1.)
            draw.AddRectFilled(tuple(point-2),tuple(point+size+2),0xDD251A12)
            draw.AddText(tuple(point),0xFFFFFFFF,label)
        ui.TextColored((.57,.66,.75,1),'294 beams | colors identify panels')
        # Orbit changes visibility, not the selector's layout height.
        ui.Text('Selected beam is on the far side.' if b.selected not in self.visible_tiles else ' ')
        if not compact:ui.End()
