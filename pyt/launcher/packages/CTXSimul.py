"""
 Support for multiple compilations of the module (e.g., TRACE, NON_TRACE) and paths
 for the module (common un CTXHOME or local to CPWD
"""
import sys
import os
machine=os.uname()[0]

DIX=""
if 'TRACE' in os.environ.keys() and os.environ['TRACE'] == "Yes":
  DIX = "_trace"

localpwd = "%s/pyt/launcher/packages/%s%s" % (os.getcwd(),machine,DIX)

if os.path.exists(localpwd):
  print ("local path exist %s"%localpwd)
  sys.path.insert(0, localpwd)
else:
  if not 'CTXHOME' in os.environ:
    raise Exception("$CTXHOME is not set")
  CTX=os.environ['CTXHOME']
  print("using remote path for library")
  print('%s/pyt/launcher/packages/%s%s'%(CTX,machine, DIX))
  sys.path.insert(0, '%s/pyt/launcher/packages/%s%s'%(CTX,machine, DIX))
import CTXSwig
