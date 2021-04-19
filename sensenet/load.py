import sensenet.importers
np = sensenet.importers.import_numpy()
tf = sensenet.importers.import_tensorflow()

from sensenet.constants import NUMERIC, CATEGORICAL, IMAGE_PATH
from sensenet.constants import STRING_INPUTS, NUMERIC_INPUTS

def load_points(preprocessors, points):
    rows = len(points)
    cols = len(preprocessors)

    inputs = {
        NUMERIC_INPUTS: np.zeros((rows, cols), dtype=np.float32),
        STRING_INPUTS: [list() for _ in range(rows)]
    }

    for i, proc in enumerate(preprocessors):
        pidx = proc['index']
        values = proc.get('values', None)

        if proc['type'] == NUMERIC:
            for j, p in enumerate(points):
                inputs[NUMERIC_INPUTS][j, i] = float(p[pidx])
        elif proc['type'] in [IMAGE_PATH, CATEGORICAL]:
            for j, p in enumerate(points):
                inputs[STRING_INPUTS][j].append(str(p[pidx]))
        else:
            raise ValueError('Unknown processor type "%s"' % proc['type'])

    inputs[STRING_INPUTS] = np.array(inputs[STRING_INPUTS])

    return inputs
