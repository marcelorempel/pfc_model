"""
This script defines auxiliar functions used in other modules.

It contains:
    set_working_dir: a function that sets the working directory.
    
    set_simulation_dir: a function that sets the directory intended to
    save simulation results.
    
    time_report: a decorator that measures the duration of another
    function execution.
    
    BaseClass: a dataclass which other classes in other modules are
    subclassed from.
    
    remove_read_only: a function to bypass erros in shutil.rmtree due
    to read-only setting.
"""

import os
import time as tm
from dataclasses import dataclass
import stat

__all__ = ['set_working_dir', 'set_simulation_dir', 'time_report', 'BaseClass',
           'remove_read_only']

def set_working_dir(name):
    """Set working directory."""
    
    os.chdir(name)

def set_simulation_dir(name=None):
    """Set the directory where simulation results can be saved. If
    a directory with the requested name already exists, '_{}' is
    appended to the name ({} is the lowest integer so that the
    resulting name does not correspond to an existing directory).
    
    Parameter
    ---------
    name: str, optional
        Name of the path. If not given, the default name 
        'Simulation_{}' is set ({} is the lowest integer so 
        that the resulting name does not correspond to an 
        existing directory).
        
    Returns
    -------
    out: str
        Resulting path name.   
    """
    if name is None:
        num = 0
        name = 'Simulation_{}'
        while os.path.isdir(name.format(num)):
            num+=1
        name=name.format(num)
    else:
        file_name = name
        num = 1
        while os.path.exists(file_name):
            file_name = name + '_{}'.format(num)
            num += 1
        name = file_name
    
    os.mkdir(name)
    print('REPORT: Directory created:', name, end='\n\n')
    
    return name

def time_report(text=None):
    """ Decorator that measures execution duration of the wrapped
    function. The elapses time is printed to the screen.
    
    Parameter
    ---------
    text: str, optional
        Function alias to be printed to screen. It not given, __name__
        is used.         
    
    Returns
    -------
    out: function
        Wrapped function.    
    """
    
    def wrap(f):    
        def wrapped_f(*args, **kwargs):
            start_time = tm.time()
            print_out1 = '|   TIME REPORT:   |'
            print_out2 =  ' Starting {} ...'.format(text if text is not None
                                                    else f.__name__)
            print('-'*len(print_out1) + ' '*len(print_out2))
            print(print_out1+print_out2)
            print('-'*len(print_out1) + ' '*len(print_out2), end='\n\n')
            ret = f(*args, **kwargs)
            elapsed = tm.time()-start_time
            if elapsed>=3600:
                print_out = ('||   TIME REPORT: {} completed '
                    .format(text if text is not None else f.__name__)
                    + 'in {:.0f} h, {:.0f} min and {:.2f} s.   ||'
                    .format(elapsed//3600, (elapsed%3600)//60, elapsed%60)
                    )
                print('-'*len(print_out))
                print('|'+'-'*(len(print_out)-2)+'|')
                print(print_out)
            elif elapsed>=60:
                print_out = ('||   TIME REPORT: {} completed in '
                             .format(text if text is not None else f.__name__)
                             + '{:.0f} min and {:.2f} s.   ||'
                             .format(elapsed//60, elapsed%60)
                )
                print('-'*len(print_out))
                print('|'+'-'*(len(print_out)-2)+'|')
                print(print_out)
            else:
                print_out = ('||   TIME REPORT: {} completed in {:.2f} s.   ||'
                .format(text if text is not None else f.__name__, elapsed))
                print('-'*len(print_out))
                print('|'+'-'*(len(print_out)-2)+'|')
                print(print_out)
            print('|'+'-'*(len(print_out)-2)+'|')
            print('-'*len(print_out), end='\n\n')
            return ret
        return wrapped_f
    return wrap

@dataclass
class BaseClass:
    """A dataclass used as superclass in other modules. This class
    allows attributes to be accessed as dictionary values by 
    subscription.
    
    Methods
    -------
    tree: show nested BaseClasses.
    """
    
    def tree(self, max_depth=None):
        """Show data that are stored as nested BaseClasses.
        
        Parameter
        ---------
        max_depth: int, optional
            Maximum depth of nested BaseClasses that is showed. If
            not given, all nested BaseClasses are shown.      
        """
              
        if len(self._values()) == 0:
            print('{}'.format(self.__class__.__name__))
        else:
            print('{} instance containing:'.format(self.__class__.__name__))       
            for key in  self._keys():
                if (getattr(self[key], '_subtree', None) is not None 
                        and (max_depth is None or max_depth > 0)):
                    post = ' instance containing:'
                else:
                    post = ''
                print(' ' * 4 + '|- {}: {}'.format(
                    key, self[key].__class__.__name__) + post)
                if getattr(self[key], '_subtree', None) is not None:
                    if max_depth is None or max_depth > 0:
                        self[key]._subtree(1, max_depth)    
    
    def _keys(self):
        return list(self.__dict__.keys())
    
    
    def _values(self):
        return list(self.__dict__.values())
    
    def _subtree(self, depth, max_depth):
        tab = '    |' * depth
       
        for key in  self._keys():
            if (getattr(self[key], '_subtree', None) is not None 
                and (max_depth is None or max_depth > depth)):
                post = ' instance containing:'
            else:
                post = ''
            print(tab + ' '*4 + '|- {}: {}'.format(
                key, self[key].__class__.__name__)+post
                )
            if getattr(self[key], '_subtree', None) is not None:
                if max_depth is None or max_depth > depth:
                    self[key]._subtree(depth+1, max_depth)
    
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, item, value):
        setattr(self, item, value)
    
def remove_read_only(func, path, excinfo):
    # Using os.chmod with stat.S_IWRITE to allow write permissions
    os.chmod(path, stat.S_IWRITE)
    func(path)
    
