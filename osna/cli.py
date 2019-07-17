# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""
import click
import glob
import sys

from . import credentials_path, config

@click.group()
def main(args=None):
    """Console script for osna."""
    return 0

@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True, default=credentials_path, help='a json file of twitter tokens')
def web(twitter_credentials):
	from .app import app
	app.run(host='0.0.0.0', debug=True)
	

@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
	"""
	Read all files in this directory and its subdirectories and print statistics.
	"""
	print('reading from %s' % directory)
	# use glob to iterate all files matching desired pattern (e.g., .json files).
	# recursively search subdirectories.

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
