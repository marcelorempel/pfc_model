import os
import time as tm
from dataclasses import dataclass

__all__ = ['set_simulation_dir', 'time_report', 'BaseClass']

def set_simulation_dir(name=None):
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
    print('REPORT: simulation_dir created:', name, end='\n\n')
    
    return name

def time_report(text=None):
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
    pass

    def view(self, item=None):
        self.tree(0, item)
    
    def tree(self, max_depth=None, item=None):
        if item is None:
            inst=self
        else:
            inst=self[item]
      
        
        if len(inst.values())==0:
            print('{}'.format(inst.__class__.__name__))
        else:
            print('{} instance containing:'.format(inst.__class__.__name__))       
            for key in  inst.keys():
                if getattr(inst[key], 'subtree', None) is not None and (max_depth is None or max_depth>0):
                    post = ' instance containing:'
                else:
                    post = ''
                print(' '*4+'|- {}: {}'.format(key, inst[key].__class__.__name__)+post)
                # print(max_depth)
                if getattr(inst[key], 'subtree', None) is not None:
                    if max_depth is None or max_depth>0:
                        inst[key].subtree(1, max_depth)    
    
    def subtree(self, depth, max_depth):
        tab = '    |' *depth
       
        for key in  self.keys():
            if (getattr(self[key], 'subtree', None) is not None 
                and (max_depth is None or max_depth>depth)):
                post = ' instance containing:'
            else:
                post = ''
            print(tab+ ' '*4+'|- {}: {}'.format(
                key, self[key].__class__.__name__)+post
                )
            if getattr(self[key], 'subtree', None) is not None:
                if max_depth is None or max_depth>depth:
                    self[key].subtree(depth+1, max_depth)

    def keys(self):
        return list(self.__dict__.keys())
    
    
    def values(self):
        return list(self.__dict__.values())
    
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, item, value):
        setattr(self, item, value)
    