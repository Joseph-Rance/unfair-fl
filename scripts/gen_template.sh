#!/bin/bash
cat configs/templates/$1.yaml <(echo) configs/templates/$2.yaml \
                              <(echo) configs/templates/$3.yaml > configs/gen_config.yaml