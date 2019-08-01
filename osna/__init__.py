# -*- coding: utf-8 -*-

"""Top-level package for elevate-osna."""

__author__ = """A Student"""
__email__ = 'student@example.com'
__version__ = '0.1.0'

# -*- coding: utf-8 -*-
import configparser
import os

# ~/.osna/osna.cfg will contain configuration information for the project,
# such as where data will be downloaded from.
def write_default_config(path):
	w = open(path, 'wt')
	w.write('[data]\n')
	w.write('data = https://www.dropbox.com/s/...')  # to be determined
	w.close()

# Find OSNA_HOME path
if 'OSNA_HOME' in os.environ:
    osna_path = os.environ['OSNA_HOME']
else:
    osna_path = os.environ['HOME'] + os.path.sep + '.osna' + os.path.sep

# Make osna directory if not present
try:
    os.makedirs(osna_path)
except:
    pass

# main config file.
config_path = osna_path + 'osna.cfg'
# twitter credentials.
credentials_path = osna_path + 'credentials.json'
# classifier
clf_path = osna_path + 'clf.pkl'

# write default config if not present.
if not os.path.isfile(config_path):
    write_default_config(config_path)

# config variable now accessible throughout project.
config = configparser.RawConfigParser()
config.read(config_path)