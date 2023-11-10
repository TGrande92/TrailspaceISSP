import socket
import time
import logging
from warnings import warn
import os

fgclient_logger = logging.getLogger('fgclient')

class FgClient:

  def __init__(self,host='127.0.0.1',port=5051,savelog=True):
    self._logger = None
    if savelog:
      self.init_logger()
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect((host,port))
    self.term = bytes([13,10])
    msg = b'data'+self.term
    self.sock.sendall(msg)
    self._tic = None

  def init_logger(self):
    self._logger = logging.getLogger('fgclient')
    self._logger.setLevel(logging.INFO)
    self._logger.handlers = [] # turn off hanging files
    if not os.path.isdir('logs'):
      os.mkdir('logs')
    filehandler = logging.FileHandler('logs/fglog'+time.strftime('%y%m%d%H%M%S')+'.csv')
    self._logger.addHandler(filehandler)


  def _get_prop(self,prop_name):
    msg = bytes('get '+prop_name,encoding='utf8')+self.term
    self.sock.sendall(msg)
    data = self.sock.recv(1024)
    if self._logger:
      self._logger.debug('{},{},{},G'.format(time.time(),prop_name,str(data)))
    return(data)

  def get_prop_str(self,prop_name):
    return(str(self._get_prop(prop_name)))

  def get_prop_float(self,prop_name):
    res = float(self._get_prop(prop_name))
    if self._logger:
      self._logger.info('{},{},{},G'.format(time.time(),prop_name,res))
    return(res)

  def set_prop(self,prop_name,new_value):
    st = 'set {} {}'.format(prop_name,new_value)
    msg = bytes(st,encoding='utf8')+self.term
    self.sock.sendall(msg)
    if self._logger:
      self._logger.info('{},{},{},S'.format(time.time(),prop_name,new_value))

  def log_entry(self,log_name,value):
    if self._logger:
      self._logger.info('{},{},{},L'.format(time.time(),log_name,value))

  # def vertical_speed_fps(self):
  #   return(self.get_prop_float('/velocities/vertical-speed-fps'))

  # def heading_deg(self):
  #   return(self.get_prop_float('/orientation/heading-deg'))

  def altitude_ft(self):
    return(self.get_prop_float('/position/altitude-ft'))

  def get_xaccel(self):
    return(self.get_prop_float('/accelerations/pilot/x-accel-fps_sec'))

  def get_yaccel(self):
    return(self.get_prop_float('/accelerations/pilot/y-accel-fps_sec'))

  def get_zaccel(self):
    return(self.get_prop_float("/accelerations/pilot/z-accel-fps_sec"))
  def elapsed_time(self):
    return(self.get_prop_float('/sim/time/elapsed-sec'))
  def get_windspeed(self):
    return(self.get_prop_float('/environment/wind-speed-kt'))

  def get_wind_direction(self):
    return (self.get_prop_float('/environment/wind-from-heading-deg'))

  def get_elevator(self):
    return(self.get_prop_float('/controls/flight/elevator'))
  
  def get_aileron(self):
    return(self.get_prop_float('/controls/flight/aileron'))
    
  def get_rudder(self):
    return(self.get_prop_float('/controls/flight/rudder'))
  
  # def get_elapsed_time(self):
  #   return(self.get_prop_float('/sim/time/elapsed-sec'))
  
  def set_elevator(self,val):
    self.set_prop('/controls/flight/elevator',val)

  def set_windspeed(self,val):
    self.set_prop('/environment/wind-speed-kt',val)

  def set_wind_direction(self,val):
    self.set_prop('/environment/wind-from-heading-deg',val)
  
  def set_aileron(self,val):
    self.set_prop('/controls/flight/aileron',val)
  
  def set_rudder(self,val):
    self.set_prop('/controls/flight/rudder',val)

  def set_throttle(self,val):
    self.set_prop('/controls/engines/engine/throttle',val)