try:
    from tqdm import tqdm
except ImportError:
    def tqdm(f, desc=None):
        return f
