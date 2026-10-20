"""Lightweight live path-gain chart; no figure rasterization or texture upload."""
import numpy as np
import polyscope.imgui as ui


def draw_rss_scale(app):
    from matplotlib import colormaps
    cfg=app.cfg.radio_map;key=(cfg.color_map,cfg.vmin,cfg.vmax)
    cache=getattr(app,'_rss_scale',None)
    if cache is None or cache[0]!=key:
        colors=(colormaps[cfg.color_map](np.linspace(0,1,64))*255).astype(int)
        packed=[int(r)|(int(g)<<8)|(int(b)<<16)|(255<<24) for r,g,b,a in colors]
        app._rss_scale=cache=(key,packed)
    ui.TextColored((.57,.66,.75,1),'RSS (dBm)  /  weak to strong')
    width=ui.GetContentRegionAvail()[0];font=ui.GetFontSize()
    x,y=ui.GetCursorScreenPos();ui.InvisibleButton('##rss_scale',(width,font+25))
    draw=ui.GetWindowDrawList()
    for i,color in enumerate(cache[1]):
        draw.AddRectFilled((x+width*i/64,y),(x+width*(i+1)/64+1,y+14),color)
    for fraction,value in ((0.,cfg.vmin),(.5,(cfg.vmin+cfg.vmax)/2),(1.,cfg.vmax)):
        label=f'{value:g}';offset=0 if fraction==0 else len(label)*font*.6 if fraction==1 else len(label)*font*.3
        draw.AddText((x+width*fraction-offset,y+18),0xFFEAE5DE,label)


def draw_path_chart(app):
    powers=app.demo_path_powers;delays=app.demo_path_delays
    cache=getattr(app,'_live_chart',None)
    if cache is None or cache['powers'] is not powers or cache['delays'] is not delays:
        valid=np.isfinite(powers)&(powers>0)&np.isfinite(delays)
        gain=10*np.log10(powers[valid]);delay=delays[valid]
        count=len(gain)
        if count>100:
            strongest=np.argsort(gain)[-100:];gain=gain[strongest];delay=delay[strongest]
        low=min(-40.,float(np.floor(gain.min()/20)*20)) if len(gain) else -120.
        high=max(low+20.,float(np.ceil(gain.max()/20)*20)) if len(gain) else 0.
        xmax=max(1.,float(delay.max())*1.05) if len(delay) else 1.
        cache=dict(powers=powers,delays=delays,gain=gain,delay=delay,low=low,high=high,xmax=xmax,count=count)
        app._live_chart=cache
    ui.TextColored((.57,.66,.75,1),'Path gain (dB) / delay (ns)')
    width=ui.GetContentRegionAvail()[0];font=ui.GetFontSize();height=max(145.,width*.60)
    start=np.asarray(ui.GetCursorScreenPos());ui.InvisibleButton('##live_path_chart',(width,height))
    draw=ui.GetWindowDrawList();end=start+[width,height]
    draw.AddRectFilled(tuple(start),tuple(end),0xFF30251D)
    left=start[0]+font*2.7;right=end[0]-font*.5
    top=start[1]+font*.6;bottom=end[1]-font*1.7
    for fraction in (0.,.5,1.):
        y=bottom-fraction*(bottom-top)
        draw.AddLine((left,y),(right,y),0xFF46515C,1.)
        value=cache['low']+fraction*(cache['high']-cache['low'])
        draw.AddText((start[0]+3,y-font*.5),0xFFC6B9AB,f'{value:.0f}')
    for fraction in (0.,.5,1.):
        x=left+fraction*(right-left)
        text=f'{cache["xmax"]*fraction:.0f}'
        draw.AddText((min(x,end[0]-len(text)*font*.6),bottom+5),0xFFC6B9AB,text)
    for delay,gain in zip(cache['delay'],cache['gain']):
        x=left+delay/cache['xmax']*(right-left)
        y=bottom-(gain-cache['low'])/(cache['high']-cache['low'])*(bottom-top)
        draw.AddLine((x,bottom),(x,y),0xFF66C7F2,2.)
        draw.AddCircleFilled((x,y),2.5,0xFF66C7F2)
    if not cache['count']:draw.AddText((left,top+font),0xFFC6B9AB,'No valid paths')
    if cache['count']>100:ui.TextColored((.57,.66,.75,1),'Showing the strongest 100 paths.')
