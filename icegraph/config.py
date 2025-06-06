# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
import yaml


# CONSTANTS

PROGRAM_NAME = "icegraph"

# PATHS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")


PASS_2_GCD_PATH = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_IC86.All_Pass2.i3.gz"

# TEMP PATHS

TEST_I3_DIR = "/data/i3store/users/tstjean/i3_10_test"
TEST_H5_DIR = "/data/i3store/users/tstjean/hdf5_10_test"
TEST_H5_FILE = "/data/i3store/users/tstjean/data/data.hdf5"

# CONVERTER CONFIG

CHUNK_SIZE = 100000

# EXTRACTOR CONFIG

EXTRACTOR_OUTDIR_NAME = "extraction"

# TEMP VALUES TO BE TRANSFERRED TO CONFIG FILE

PULSE_SERIES = "InIceDSTPulses"

FEATURES_TABLE_NAME = "ml_suite_features"
TRUTH_TABLE_NAME = "classification"

MCTREE_NAME = "I3MCTree_preMuonProp"
WEIGHT_DICT_NAME = "I3MCWeightDict"
BG_MCTREE_NAME = "BackgroundI3MCTree"
