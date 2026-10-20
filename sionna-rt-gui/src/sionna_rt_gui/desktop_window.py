"""Maximized decorated window for this process; native caption buttons remain."""
import os
import ctypes as c
from ctypes import wintypes as w


class FullscreenWindow:
    def __init__(self):
        self.active=False;self.hwnd=None
        if os.name!='nt':return
        self.user=c.WinDLL('user32',use_last_error=True)
        self.user.GetWindowLongPtrW.argtypes=[w.HWND,c.c_int];self.user.GetWindowLongPtrW.restype=c.c_ssize_t
        self.user.SetWindowLongPtrW.argtypes=[w.HWND,c.c_int,c.c_ssize_t];self.user.SetWindowLongPtrW.restype=c.c_ssize_t
        self.user.GetWindowRect.argtypes=[w.HWND,c.POINTER(w.RECT)]
        self.user.GetWindowThreadProcessId.argtypes=[w.HWND,c.POINTER(w.DWORD)]
        self.user.GetClassNameW.argtypes=[w.HWND,w.LPWSTR,c.c_int]
        self.user.MonitorFromWindow.argtypes=[w.HWND,w.DWORD];self.user.MonitorFromWindow.restype=w.HANDLE
        self.user.GetMonitorInfoW.argtypes=[w.HANDLE,c.c_void_p]
        self.user.SetWindowPos.argtypes=[w.HWND,w.HWND,c.c_int,c.c_int,c.c_int,c.c_int,w.UINT]
        callback=c.WINFUNCTYPE(w.BOOL,w.HWND,w.LPARAM)
        largest=[0]
        def find(hwnd,_):
            pid=w.DWORD();self.user.GetWindowThreadProcessId(hwnd,c.byref(pid))
            name=c.create_unicode_buffer(256);self.user.GetClassNameW(hwnd,name,256)
            if pid.value==os.getpid() and 'GLFW' in name.value:
                # GLFW also owns a tiny hidden helper window: select the actual viewport.
                rect=w.RECT();self.user.GetWindowRect(hwnd,c.byref(rect))
                area=(rect.right-rect.left)*(rect.bottom-rect.top)
                if area>largest[0]:self.hwnd=hwnd;largest[0]=area
            return True
        self.user.EnumWindows(callback(find),0)

    def toggle(self):
        if not self.hwnd:return False
        self.user.ShowWindow.argtypes=[w.HWND,c.c_int]
        self.user.IsZoomed.argtypes=[w.HWND];self.user.IsZoomed.restype=w.BOOL
        maximized=bool(self.user.IsZoomed(self.hwnd))
        self.user.ShowWindow(self.hwnd,9 if maximized else 3)  # SW_RESTORE / SW_MAXIMIZE
        self.active=not maximized
        return True



def layout(app):
    import polyscope as ps
    width,height=ps.get_window_size();s=app.ui_scale
    left=min(440*s,width*.34);top=max(240,height*.60)
    return left,top,top+22*s,max(180,height-top-34*s)
