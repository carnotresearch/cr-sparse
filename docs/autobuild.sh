#!/bin/bash
sphinx-autobuild --host=0.0.0.0 --port=9100 -N . _build/html --watch ../src/cr/sparse

