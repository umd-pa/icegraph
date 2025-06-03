# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os


# PATHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# TEMP PATHS

TEST_I3_DIR = "/data/i3store/users/tstjean/i3tests"
TEST_H5_DIR = "/data/i3store/users/tstjean/hdf5_tests/"
TEST_H5_FILE = "/data/i3store/users/tstjean/data/data.hdf5"

# CONVERTER CONFIG

CHUNK_SIZE = 100000

# TEMP VALUES TO BE TRANSFERRED TO CONFIG FILE

PULSE_SERIES = "InIceDSTPulses"
TRUTH_TABLE = "I3MCTree_preMuonProp"