import os
import json

BYTE_ORDER = 'big' # Meaningless; we're only going to use 1 byte
GUARD = 'bigml_tensorflow_saved_model_bundle'.encode('utf8')
HEAD_BLOCK_SIZE = 1024
BUNDLE_EXTENSION = '.smbundle'

def write_to_bundle(adir, is_header, fout):
    all_files = [c for c in os.walk(adir)]
    rel_root = None
    file_header = []

    for root, dirs, files in all_files:
        outfiles = []

        if rel_root is None:
            rel_root = root

        for afile in files:
            apath = os.path.join(root, afile)
            fsize = os.path.getsize(apath)
            outfiles.append([afile, fsize])

            if not is_header:
                with open(apath, 'rb') as fin:
                    data = fin.read()
                    fout.write(data)

                    assert fsize == len(data)

        file_header.append([os.path.relpath(root, rel_root), dirs, outfiles])

    if is_header:
        head_json = json.dumps(file_header)
        nblocks = (len(head_json) // HEAD_BLOCK_SIZE) + 1

        if nblocks > 255:
            raise ValueError('Header length is %d > 255 blocks' % nblocks)

        block_count = nblocks.to_bytes(1, BYTE_ORDER)
        head_len = nblocks * HEAD_BLOCK_SIZE
        head_bytes = block_count + head_json.ljust(head_len).encode('utf8')

        assert len(head_bytes) == head_len + 1

        fout.write(GUARD)
        fout.write(head_bytes)

def write_bundle(adir):
    assert os.path.isdir(adir)
    outpath = adir + BUNDLE_EXTENSION

    with open(outpath, 'wb') as fout:
        write_to_bundle(adir, True, fout)
        write_to_bundle(adir, False, fout)

    return outpath

def read_bundle(afile):
    top_root = afile[:-len(BUNDLE_EXTENSION)]

    with open(afile, 'rb') as fin:
        guard = fin.read(len(GUARD))

        if guard != GUARD:
            raise ValueError('Guard bytes are %s' % guard)

        head_blocks = int.from_bytes(fin.read(1), BYTE_ORDER)
        head_bytes = fin.read(head_blocks * HEAD_BLOCK_SIZE)
        header = json.loads(head_bytes.decode('utf8'))

        for root, dirs, files in header:
            root_path = os.path.join(top_root, root)
            os.makedirs(root_path, exist_ok=True)

            for adir in dirs:
                os.makedirs(os.path.join(root_path, adir), exist_ok=True)

            for outfile, filebytes in files:
                apath = os.path.join(root_path, outfile)
                data = fin.read(filebytes)

                with open(apath, 'wb') as fout:
                    fout.write(data)

    return top_root
