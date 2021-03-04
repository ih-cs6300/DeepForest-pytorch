#test callbacks
from deepforest import main
from deepforest import callbacks
import glob
import os
import pytest

from deepforest import get_data
from .conftest import download_release

@pytest.fixture()
def m(download_release):
    m = main.deepforest()
    m.config["train"]["csv_file"] = get_data("example.csv") 
    m.config["train"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    m.config["train"]["fast_dev_run"] = False
    m.config["batch_size"] = 2
       
    m.config["validation"]["csv_file"] = get_data("example.csv") 
    m.config["validation"]["root_dir"] = os.path.dirname(get_data("example.csv"))
    
    m.use_release()
    
    return m

def test_log_images(m, tmpdir):
    im_callback = callbacks.images_callback(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=tmpdir)
    m.create_trainer(callbacks=[im_callback])
    m.max_steps = 2
    m.trainer.fit(m)
    saved_images = glob.glob("{}/*.png".format(tmpdir))
    assert len(saved_images) == 1
    
    
    
    