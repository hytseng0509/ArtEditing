import sys
from subprocess import call

if len(sys.argv) != 2:
    raise Exception('Incorrect command! python process.py DATASET [face, anime]')
dataset = sys.argv[1]

print('Download {} dataset'.format(dataset))

filename = dataset + '.tar.gz'
call('wget http://vllab.ucmerced.edu/ym41608/projects/ArtEditing/data/' + filename, shell=True)
call('tar -xzf ' + filename, shell=True)
call('rm -rf ' + filename, shell=True)
