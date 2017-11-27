import os

root = os.getcwd()
afrpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CONFIGS = (
    'PATHDB',
    'PATHEXPORTS',
    'PATHTBI'
)

os.chdir(os.path.join(afrpath, 'config'))
for filename in os.listdir():
    if filename.lower() == 'config.ini':
        with open('config.ini') as file:
            cfg = file.read().strip().splitlines()
    elif filename.lower() == 'sets.ini':
        with open('sets.ini') as file:
            sets = file.read().strip().splitlines()

content = [
    'import os',
    'WIDTH=0',
    'HEIGHT=1',
    'PFX=2',
    'SFX=3',
    'INIT=4',
    'END=5',
    'PATH=6',
    'NAMES=7'
]
for line in cfg:
    line = line.strip()
    if line and line[0] not in '#;':
        try:
            temp = line.split('=')
            key = temp[0].upper()
            if key in CONFIGS:
                value = f"os.path.normpath(r'{temp[1]}')"
                content.append(f'{key}={value}')
        except:
            pass
d = []
setd = ''
setkv = []
for line in sets:
    line = line.strip()
    if line and line[0] not in '#;':
        if line[0] == '[':
            if setd:
                setd += ','.join(setkv) + '}'
                d.append(setd)
                setd = f"'{line[1:-1]}':{{"
                setkv = []
            else:
                setd = f"'{line[1:-1]}':{{"
        else:
            temp = line.split('=')
            key = temp[0].upper()
            if key == 'PATH':
                value = f"os.path.normpath(r'{temp[1]}')"
            elif key == 'NAMES':
                if not temp[1]:
                    value = '[]'
                else:
                    value = '[' + ",".join([f"'{name.strip()}'" for name in temp[1].split(',')]) + ']'
            elif key in ('PFX', 'SFX'):
                value = f"'{temp[1]}'"
            else:
                # key is WIDTH, HEIGHT, INIT, or END
                value = temp[1]
            setkv.append(f"{key}:{value}")
setd += ','.join(setkv) + '}'
d.append(setd)
content.append('SETS={' + ','.join(d) + '}')

os.chdir(os.path.join(afrpath, 'prereq'))
with open('consts.py', 'w') as file:
    file.write('\n'.join(content))
os.chdir(root)