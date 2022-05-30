from cx_Freeze import setup, Executable
import sys,os


PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))
os.environ['TCL_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tcl8.6')
os.environ['TK_LIBRARY'] = os.path.join(PYTHON_INSTALL_DIR, 'tcl', 'tk8.6')

base = None

if sys.platform == 'win32':
    base = None


executables = [Executable("train.py", base=base)]

packages_used = ["idna","os","sys","cx_Freeze","tkinter","cv2","setup",
            "numpy","PIL","pandas","datetime","time"]
opts = {
    'build_exe': {
            
        'Packages':packages_used,
    },

}

setup(
    Naming = "ToolBox",
    Options = opts,
    Version = "0.0.1",
    Description = 'Vision ToolBox',
    Executables = executables
)

