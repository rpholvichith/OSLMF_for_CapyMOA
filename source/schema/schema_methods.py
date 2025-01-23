import numpy as np

from capymoa.base import Schema
from typing import Tuple

def from_schema_to_ord_cont(schema: Schema) -> Tuple[np.ndarray, np.ndarray]:
    n_features = schema.get_moa_header().numAttributes()-1

    all_cont_indices = np.zeros(n_features).astype(bool)
    all_ord_indices = np.zeros(n_features).astype(bool)

    for i in range(n_features):
        attribute_i = schema.get_moa_header().attribute(i)
        if attribute_i.isNominal():
            all_ord_indices[i] = True
        elif attribute_i.isNumeric():
            all_cont_indices[i] = True
        else:
            raise Exception(f"The attribute {attribute_i.name()} of data stream {schema.dataset_name} is neither NOMINAL nor NUMERIC.")

    return all_ord_indices, all_cont_indices
