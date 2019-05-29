#! /usr/bin/env python

__title__ = 'sfindernn'
__version__ = '1.0.0'
__author__ = 'Simone Riggi'
__license__ = 'GPL3'
__date__ = '2019-05-29'
__copyright__ = 'Copyright 2019 by Simone Riggi - INAF'


import logging
import logging.config


# Create the Logger
logging.basicConfig(format="%(module)s:%(levelname)s %(message)s")
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("This is sfindernn {0}-({1})".format(__version__, __date__))


