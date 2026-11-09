"""Run InstantHMR's landmark readout with only the vertices it references."""

import numpy as np
import onnxruntime as ort


class LandmarkBody:
    def __init__(self, path):
        try:
            import onnx
        except ImportError as error:
            raise SystemExit("Landmarks-only mode requires ONNX. Install it with "
                             "uv pip install --python .venv/bin/python onnx.") from error

        model = onnx.load(path)
        graph = model.graph
        mesh = next(output for output in graph.output if output.name == "vertices")
        n_vertices = mesh.type.tensor_type.shape.dim[1].dim_value
        gather, = (node for node in graph.node if node.op_type == "Gather" and node.input[0] == mesh.name)
        constants = {node.output[0]: node.attribute[0].t for node in graph.node if node.op_type == "Constant"}
        indices = onnx.numpy_helper.to_array(constants[gather.input[1]])
        n_selected = len(indices)

        # The released body graph skins four influences per vertex. Apply the
        # landmark Gather to the independent skinning inputs instead of its output.
        replacements = 0
        for name, tensor in constants.items():
            data = onnx.numpy_helper.to_array(tensor)
            if name == gather.input[1]:
                sliced = np.arange(n_selected, dtype=data.dtype)
            elif data.shape == (45, n_vertices * 3):
                sliced = data.reshape(45, n_vertices, 3)[:, indices].reshape(45, -1)
            elif data.shape in ((n_vertices, 3), (n_vertices, 4, 1)):
                sliced = data[indices]
            elif data.shape == (n_vertices * 4,):
                sliced = data.reshape(n_vertices, 4)[indices].reshape(-1)
            elif data.shape == (2,) and data.dtype == np.int64 and data[0] == n_vertices and data[1] in (3, 4):
                sliced = np.array([n_selected, data[1]], dtype=data.dtype)
            else:
                continue
            tensor.CopyFrom(onnx.numpy_helper.from_array(sliced))
            replacements += 1
        if replacements != 7:
            raise SystemExit("Landmarks-only mode expects the released InstantHMR 1.1 body graph.")

        graph.output.remove(mesh)
        onnx.checker.check_model(model)
        self.session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    def run(self, _outputs, feeds):
        joints, = self.session.run(["joints_3d"], feeds)
        # The upstream pipeline unpacks both body results for each person.
        return joints, [None] * len(joints)
