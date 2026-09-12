import numpy as np
from omnimira import from_pretrained

def test_bundled_epoch1000_extracts_named_roi_features():
    extractor=from_pretrained("cpu")
    features=extractor.extract_array(np.zeros((96,112,96),dtype=np.float32),"t1")
    assert {name:value.shape for name,value in features.items()}=={"aal3":(166,128),"ho69":(69,128),"yeo7":(7,128)}
    assert all(np.isfinite(value).all() for value in features.values())
