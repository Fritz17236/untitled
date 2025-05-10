import pytest
import h5py 
from pathlib import Path 
from tempfile import NamedTemporaryFile
import numpy as np 
import nonlinear_approximator as na

DB_RTILDE = 'Rtilde'
DB_Q2 = 'Q2'
DB_QTB = "QTB"
DB_DECODERS = 'decoders'

@pytest.fixture
def dummy_config(tmp_path): 
    return na.params.RegressionParams(
        width=1000,
        depth=50,
        input_dimension=10,
        transform_type=na.activations.TransformType.TENT,
        transform_params=na.params.TentParams(mu=1.99),
        output_dimension=2,
        neuron_chunk_size=100,
        storage_path=tmp_path.joinpath('test_storage.hdf5'),
        batch_size=1000, 
    )

def init_hdf5(config: na.params.RegressionParams):
    with h5py.File(config.storage_path, 'a') as file:
        file.create_dataset(DB_RTILDE, shape=(config.depth, config.depth))
        file.create_dataset(DB_QTB, shape=(config.depth, config.output_dimension))
        file.create_dataset(DB_Q2, SHAPE=(10 * config.depth, config.depth), maxshape=(None, config.depth))
        file.create_dataset(DB_DECODERS, shape=(config.depth, config.output_dimension, config.width))
    
def test_initialize_qr_close_to_pinv_comp(dummy_config):
        assert dummy_config.storage_path
        print("config has storage path: ", dummy_config.storage_path)
        A = np.random.random((NUM_SAMPLES:= 1000, dummy_config.depth, dummy_config.width))[:, :, 0] # test neuron at index 0
        B = np.ones((NUM_SAMPLES, dummy_config.output_dimension))
        
        Rtilde, Q2, QTB, Q1, R1 = na.training.qr_initialize(A, B)
        assert Rtilde.shape == (dummy_config.depth, dummy_config.depth)
        assert Q2.shape == (dummy_config.depth, dummy_config.depth)
        assert QTB.shape == (dummy_config.depth, dummy_config.output_dimension)
        assert Q1.shape == (NUM_SAMPLES, dummy_config.depth)
        assert R1.shape == (dummy_config.depth, dummy_config.depth)

        decoder = np.linalg.inv(Rtilde) @ QTB
        pinv_decoder = np.linalg.pinv(A) @ B
        assert np.isclose(
            np.linalg.norm(A @ decoder - B, ord='fro'),
            np.linalg.norm(A @ pinv_decoder - B, ord='fro')
        )
        
def test_update_qr_close_to_pinv_comp(dummy_config):
        assert dummy_config.storage_path
        print("config has storage path: ", dummy_config.storage_path)
        A = np.random.random((NUM_SAMPLES:= 1000, dummy_config.depth, dummy_config.width))[:, :, 0] # test neuron at index 0
        B = np.ones((NUM_SAMPLES, dummy_config.output_dimension))
        
        A_new = np.random.random((NUM_SAMPLES:= 1000, dummy_config.depth, dummy_config.width))[:, :, 0] # test neuron at index 0
        B_new = 0.5 * np.ones((NUM_SAMPLES, dummy_config.output_dimension))
        
        Rtilde, Q2, QTB, Q1, R1 = na.training.qr_initialize(A, B)
        Rtilde, Q2, QTB, Q1, R1 = na.training.qr_update(A_new, B_new, Rtilde, Q2, QTB, Q1, R1)
        
        A_all = np.vstack((A, A_new))
        B_all = np.vstack((B, B_new))
        
        decoder = np.linalg.inv(Rtilde) @ QTB
        pinv_decoder = np.linalg.pinv(A_all) @ B_all
        assert np.isclose(
            np.linalg.norm(A_all @ decoder - B_all, ord='fro'),
            np.linalg.norm(A_all @ pinv_decoder - B_all, ord='fro')
        )
        
