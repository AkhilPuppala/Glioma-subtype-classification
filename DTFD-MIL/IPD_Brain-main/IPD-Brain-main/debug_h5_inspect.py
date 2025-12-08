import sys
try:
    import h5py
except Exception as e:
    print('h5py import error:', e)
    print('Please install h5py in the environment (pip install h5py)')
    sys.exit(1)

def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        try:
            print(f"DATASET: {name} shape={obj.shape} dtype={obj.dtype}")
        except Exception as e:
            print(f"DATASET: {name} (error getting shape: {e})")
    elif isinstance(obj, h5py.Group):
        print(f"GROUP: {name}")

def inspect(path):
    with h5py.File(path, 'r') as f:
        print('Root keys:', list(f.keys()))
        f.visititems(walk)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python debug_h5_inspect.py <h5file>')
        sys.exit(1)
    inspect(sys.argv[1])
