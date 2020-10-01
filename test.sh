#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_conv_vis tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_conv_vis --with-doctest